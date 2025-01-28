part of '../maid_llm.dart';

class LlamaCPP {
  static Completer? _completer;
  static SendPort? _sendPort;
  static lcpp? _lib;
  static void Function(String)? _log;

  /// Getter for the Llama library.
  ///
  /// Loads the library based on the current platform.
  static lcpp get lib {
    if (_lib == null) {
      if (Platform.isWindows) {
        _lib = lcpp(DynamicLibrary.open('llama.dll'));
      } 
      else if (Platform.isLinux || Platform.isAndroid) {
        _lib = lcpp(DynamicLibrary.open('llama.so'));
      } 
      else if (Platform.isMacOS || Platform.isIOS) {
        _lib = lcpp(DynamicLibrary.open('maid_llm.framework/maid_llm'));
      } 
      else {
        throw Exception('Unsupported platform');
      }
    }
    return _lib!;
  }

  LCPP(GptParams params, {void Function(String)? log}) {
    _log = log;

    if (_log != null) {
      _log!("Initializing LLM");
    }

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    _completer = Completer();

    Isolate.spawn(_initIsolate, (params, _sendPort!)).then((value) async {
      receivePort.listen((data) {
        if (data is int) {
          if (data == 0) {
            _completer!.complete();
          } else {
            _completer!.completeError(Exception('Failed to initialize LLM, Exception Uncaught'));
          } 
        } else if (data is String && _log != null) {
          _log!(data);
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

  static void _initIsolate((GptParams, SendPort) args) {
    final (params, sendPort) = args;
    _sendPort = sendPort;

    try {
      final ret = lib.maid_llm_model_init(params.toNative(), Pointer.fromFunction(_logOutput));
      if (ret != 0) {
        throw Exception('Failed to initialize model');
      }

      _sendPort!.send(ret);
    } catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static void _promptIsolate((List<ChatMessage>, String, SendPort) args) {
    final (messages, template, sendPort) = args;
    _sendPort = sendPort;

    try {
      final ret = lib.maid_llm_prompt(
        _toNativeChat(messages, template), 
        Pointer.fromFunction(_output), 
        Pointer.fromFunction(_logOutput)
      );

      if (ret != 0) {
        throw Exception('Failed to prompt');
      }
    } catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static Pointer<maid_llm_chat> _toNativeChat(List<ChatMessage> messages, String template) {
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

  final chat = calloc<maid_llm_chat>();
  chat.ref.messages = chatMessages;
  chat.ref.message_count = messageCount;
  chat.ref.buffer_size = bufferSize;

  if (template.isNotEmpty) {
    chat.ref.tmpl = template.toNativeUtf8().cast<Char>();
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
    lib.maid_llm_stop();
    await _completer!.future;
    return;
  }

  void clear() {
    lib.maid_llm_cleanup();
  }
}
