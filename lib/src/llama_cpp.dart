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
  SendPort sendPort
});

class LlamaCPP {
  static Completer? _completer;
  static SendPort? _sendPort;

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
        _lib = lcpp(ffi.DynamicLibrary.open('llama.so'));
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

    _logger('Initializing LLM');

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    _completer = Completer();

    final initParams = (
      modelPath: modelPath,
      modelParams: modelParams,
      contextParams: contextParams,
      samplingParams: samplingParams,
      sendPort: _sendPort!
    );

    Isolate.spawn(_initIsolate, initParams).then((value) async {
      receivePort.listen((data) {
        if (data is String) {
          _logger(data);

          _completer!.completeError(Exception(data));
        }
        else {
          _completer!.complete();
        }
      });

      await _completer!.future;
    });
  }

  Stream<String> prompt(List<ChatMessage> messages, String template) async* {   
    // Ensure initialization is complete
    await _completer?.future;
    _completer = Completer();

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    final promptParams = (
      messages: messages,
      sendPort: _sendPort!
    );

    final isolate = await Isolate.spawn(_promptIsolate, promptParams);

    await for (var data in receivePort) {
      if (data is (String, bool)) {
        final (message, done) = data;

        if (done) {
          receivePort.close();
          isolate.kill();
          _completer?.complete();
          return;
        }

        yield message;
      } else if (data is String && _log != null) {
        _log!(data);
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
      return;
    } catch (e) {
      args.sendPort.send(e.toString());
    }
  }

  static void _promptIsolate(PromptIsolateArguments args) {
    _sendPort = args.sendPort;

    try {
      final nCtx = lib.llama_n_ctx(_context!);

      ffi.Pointer<ffi.Char> formatted = calloc<ffi.Char>(nCtx);

      final template = lib.llama_model_chat_template(_model!, ffi.nullptr);

      int newContextLength = lib.llama_chat_apply_template(
        template, 
        args.messages.toNative(), 
        args.messages.length, 
        true, 
        formatted, 
        nCtx
      );

      if (newContextLength > nCtx) {
        formatted = calloc<ffi.Char>(newContextLength);
        newContextLength = lib.llama_chat_apply_template(
          template, 
          args.messages.toNative(), 
          args.messages.length, 
          true, 
          formatted, 
          newContextLength
        );
      }

      if (newContextLength < 0) {
        _sendPort!.send('Failed to apply template');
        return;
      }

      final prompt = formatted.cast<Utf8>().toDartString().substring(_contextLength);
    } catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static void _output(ffi.Pointer<ffi.Char> buffer, bool stop) {
    try {
      _sendPort!.send((buffer.cast<Utf8>().toDartString(), stop));
    } 
    catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static void _logOutput(ffi.Pointer<ffi.Char> message) {
    final logMessage = message.cast<Utf8>().toDartString();

    _sendPort!.send(logMessage);
  }

  Future<void> stop() async {
    lib.lcpp_stop();
    await _completer!.future;
    return;
  }

  void clear() {
    _contextLength = 0;
  }

  void _logger(String message) {
    if (_log != null) {
      _log!(message);
    }
  }
}
