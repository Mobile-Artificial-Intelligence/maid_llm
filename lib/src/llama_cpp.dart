part of '../lcpp.dart';

typedef InitIsolateArguments = ({
  String modelPath,
  ModelParams modelParams,
  ContextParams contextParams,
  SamplingParams samplingParams,
  SendPort sendPort
});

typedef PromptIsolateArguments = ({
  List<ChatMessage> messages,
  int contextLength,
  SendPort sendPort
});

typedef PromptResponse = ({
  String message, 
  bool done
});

class LlamaCPP {
  static Completer? _completer;
  static late SendPort _sendPort;

  static lcpp? _lib;
  static ffi.Pointer<llama_model>? _model;
  static ffi.Pointer<llama_context>? _context;
  static ffi.Pointer<llama_sampler>? _sampler;

  static int _contextLength = 0;

  static void Function(String)? _log;

  /// Getter for the Llama library.
  ///
  /// Loads the library based on the current platform.
  static lcpp get lib {
    if (_lib == null) {
      if (Platform.isWindows) {
        _lib = lcpp(ffi.DynamicLibrary.open('llama.dll'));
      } 
      else if (Platform.isLinux || Platform.isAndroid) {
        _lib = lcpp(ffi.DynamicLibrary.open('libllama.so'));
      } 
      else if (Platform.isMacOS || Platform.isIOS) {
        _lib = lcpp(ffi.DynamicLibrary.open('lcpp.framework/lcpp'));
      } 
      else {
        throw Exception('Unsupported platform');
      }
    }
    return _lib!;
  }

  LlamaCPP(String modelPath, ModelParams modelParams, ContextParams contextParams, SamplingParams samplingParams, {void Function(String)? log}) {
    _log = log;

    _log?.call('Initializing LLM');

    final receivePort = ReceivePort();

    _completer = Completer();

    final initParams = (
      modelPath: modelPath,
      modelParams: modelParams,
      contextParams: contextParams,
      samplingParams: samplingParams,
      sendPort: receivePort.sendPort
    );

    Isolate.spawn(_initIsolate, initParams).then((value) async {
      receivePort.listen((data) {
        if (data is String) {
          _log?.call(data);

          _completer!.completeError(Exception(data));
        }
        else {
          _completer!.complete();
        }
      });

      await _completer!.future;
    });
  }

  Stream<String> prompt(List<ChatMessage> messages) async* {   
    // Ensure initialization is complete
    await _completer?.future;
    _completer = Completer();

    final receivePort = ReceivePort();

    final promptParams = (
      messages: messages,
      contextLength: _contextLength,
      sendPort: receivePort.sendPort
    );

    final isolate = await Isolate.spawn(_promptIsolate, promptParams);

    await for (var data in receivePort) {
      if (data is PromptResponse) {
        if (data.done) {
          receivePort.close();
          isolate.kill();
          _completer?.complete();
          return;
        }

        yield data.message;
      } 
      else if (data is SendPort) {
        _sendPort = data;
      }
      else if (data is String) {
        _log?.call(data);
      }
    }
  }

  static void _initIsolate(InitIsolateArguments args) {
    _sendPort = args.sendPort;
    
    try {
      lib.ggml_backend_load_all();

      final modelParams = args.modelParams.toNative();
      
      _model = lib.llama_load_model_from_file(
        args.modelPath.toNativeUtf8().cast<ffi.Char>(), 
        modelParams
      );
      assert(_model != null && _model != ffi.nullptr, 'Failed to load model');

      final contextParams = args.contextParams.toNative();

      _context = lib.llama_init_from_model(_model!, contextParams);
      assert(_context != null && _context != ffi.nullptr, 'Failed to initialize context');

      final vocab = lib.llama_model_get_vocab(_model!);
      _sampler = args.samplingParams.toNative(vocab);
      assert(_sampler != null && _sampler != ffi.nullptr, 'Failed to initialize sampler');

      args.sendPort.send(null);
    } catch (e) {
      args.sendPort.send(e.toString());
    }
  }

  static void _promptIsolate(PromptIsolateArguments args) {
    _sendPort = args.sendPort;
    _completer = Completer();
    _stopListener();

    try {
      final nCtx = lib.llama_n_ctx(_context!);

      ffi.Pointer<ffi.Char> formatted = calloc<ffi.Char>(nCtx);

      final template = lib.llama_model_chat_template(_model!, ffi.nullptr);

      final messages = args.messages;

      int newContextLength = lib.llama_chat_apply_template(
        template, 
        messages.toNative(), 
        messages.length, 
        true, 
        formatted, 
        nCtx
      );

      if (newContextLength > nCtx) {
        formatted = calloc<ffi.Char>(newContextLength);
        newContextLength = lib.llama_chat_apply_template(
          template, 
          messages.toNative(), 
          messages.length, 
          true, 
          formatted, 
          newContextLength
        );
      }

      if (newContextLength < 0) {
        _sendPort.send('Failed to apply template');
        return;
      }

      final prompt = formatted.cast<Utf8>().toDartString().substring(_contextLength);

      final finalOutput = _generate(prompt);

      messages.add(ChatMessage(
        role: 'assistant',
        content: finalOutput
      ));

      _contextLength = lib.llama_chat_apply_template(
        template, 
        messages.toNative(), 
        messages.length, 
        false, 
        ffi.nullptr, 
        0
      );

      _sendPort.send((message: finalOutput, done: true));
    } catch (e) {
      _sendPort.send(e.toString());
    }
  }

  static void _stopListener() async  {
    final receivePort = ReceivePort();

    _sendPort.send(receivePort.sendPort);

    await for (var data in receivePort) {
      if (data is bool) {
        receivePort.close();
        _completer?.complete();
        return;
      }
    }
  }

  static String _generate(String prompt) {
    String finalOutput = '';

    final vocab = lib.llama_model_get_vocab(_model!);
    final isFirst = lib.llama_get_kv_cache_used_cells(_context!) == 0;

    final nPromptTokens = lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, ffi.nullptr, 0, isFirst, true);
    ffi.Pointer<llama_token> promptTokens = calloc<llama_token>(nPromptTokens);

    if (lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, promptTokens, nPromptTokens, isFirst, true) < 0) {
      _sendPort.send('Failed to tokenize prompt');
      return '';
    }

    llama_batch batch = lib.llama_batch_get_one(promptTokens, nPromptTokens);
    int newTokenId;
    while (!_completer!.isCompleted) {
      final nCtx = lib.llama_n_ctx(_context!);
      final nCtxUsed = lib.llama_get_kv_cache_used_cells(_context!);

      if (nCtxUsed + batch.n_tokens > nCtx) {
        _sendPort.send('Context size exceeded');
        break;
      }

      if (lib.llama_decode(_context!, batch) != 0) {
        _sendPort.send('Failed to decode');
        break;
      }

      newTokenId = lib.llama_sampler_sample(_sampler!, _context!, -1);

      // is it an end of generation?
      if (lib.llama_vocab_is_eog(vocab, newTokenId)) {
        break;
      }

      final buffer = calloc<ffi.Char>(256);
      final n = lib.llama_token_to_piece(vocab, newTokenId, buffer, 256, 0, true);
      if (n < 0) {
        _sendPort.send('Failed to convert token to piece');
        break;
      }

      final piece = buffer.cast<Utf8>().toDartString();
      finalOutput += piece;

      _sendPort.send((message: piece, done: false));

      final newTokenPointer = calloc<llama_token>(1);
      newTokenPointer.value = newTokenId;

      batch = lib.llama_batch_get_one(newTokenPointer, 1);
    }

    return finalOutput;
  }

  Future<void> stop() async {
    _sendPort.send(true);
    await _completer!.future;
    return;
  }

  void clear() {
    _contextLength = 0;
  }
}
