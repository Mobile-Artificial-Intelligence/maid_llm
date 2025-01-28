part of '../lcpp.dart';

typedef InitIsolateArguments = ({
  String modelPath,
  ModelParams modelParams,
  ContextParams contextParams,
  SendPort sendPort
});

class LlamaCPP {
  static Completer? _completer;
  static SendPort? _sendPort;

  static lcpp? _lib;
  static ffi.Pointer<llama_model>? _model;
  static ffi.Pointer<llama_context>? _context;

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

  LlamaCPP(String modelPath, ModelParams modelParams, ContextParams contextParams, {void Function(String)? log}) {
    _log = log;

    _logger('Initializing LLM');

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    _completer = Completer();

    final initParams = (
      modelPath: modelPath,
      modelParams: modelParams,
      contextParams: contextParams,
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

    final isolate = await Isolate.spawn(_promptIsolate, (messages, template, _sendPort!));

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

      args.sendPort.send(null);
      return;
    } catch (e) {
      args.sendPort.send(e.toString());
    }
  }

  static void _promptIsolate((List<ChatMessage>, String, SendPort) args) {
    final (messages, template, sendPort) = args;
    _sendPort = sendPort;

    try {
      final ret = lib.lcpp_prompt(
        _toNativeChat(messages, template), 
        ffi.Pointer.fromFunction(_output), 
        ffi.Pointer.fromFunction(_logOutput)
      );

      if (ret != 0) {
        throw Exception('Failed to prompt');
      }
    } catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static Pointer<lcpp_chat> _toNativeChat(List<ChatMessage> messages, String template) {
  // Allocate an array of pointers to llama_chat_message
  final chatMessages = calloc<llama_chat_message>(messages.length);

  int messageCount = 0;
  int bufferSize = 1;

  for (var i = 0; i < messages.length; i++) {
    if (messages[i].content.isNotEmpty) {
      chatMessages[i] = messages[i].toNative().ref;
      bufferSize += messages[i].content.length;
      messageCount++;
    }
  }

  final chat = calloc<lcpp_chat>();
  chat.ref.messages = chatMessages;
  chat.ref.message_count = messageCount;
  chat.ref.buffer_size = bufferSize;

  if (template.isNotEmpty) {
    chat.ref.tmpl = template.toNativeUtf8().cast<ffi.Char>();
  }

  return chat;
}

  static void _output(Pointer<Char> buffer, bool stop) {
    try {
      _sendPort!.send((buffer.cast<Utf8>().toDartString(), stop));
    } 
    catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static void _logOutput(Pointer<Char> message) {
    final logMessage = message.cast<Utf8>().toDartString();

    _sendPort!.send(logMessage);
  }

  Future<void> stop() async {
    lib.lcpp_stop();
    await _completer!.future;
    return;
  }

  void clear() {
    lib.lcpp_cleanup();
  }

  void _logger(String message) {
    if (_log != null) {
      _log!(message);
    }
  }
}
