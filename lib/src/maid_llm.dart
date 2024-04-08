import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:langchain/langchain.dart';
import 'gpt_params.dart';

import 'bindings.dart';

class MaidLLM {
  static Completer? _completer;
  static SendPort? _sendPort;
  static maid_llm? _lib;
  static void Function(String)? _log;

  List<ChatMessage>? _lastMessages;

  /// Getter for the Llama library.
  ///
  /// Loads the library based on the current platform.
  static maid_llm get lib {
    if (_lib == null) {
      if (Platform.isWindows) {
        _lib = maid_llm(DynamicLibrary.open('maid.dll'));
      } else if (Platform.isLinux || Platform.isAndroid) {
        _lib = maid_llm(DynamicLibrary.open('libmaid.so'));
      } else if (Platform.isMacOS || Platform.isIOS) {
        throw Exception('Unsupported platform');
        //_lib = maid_llm(DynamicLibrary.open('bin/llama.dylib'));
      } else {
        throw Exception('Unsupported platform');
      }
    }
    return _lib!;
  }

  MaidLLM(GptParams params, {void Function(String)? log}) {
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

  void reset(GptParams params) async {
    _log?.call('Resetting LLM');

    await _completer?.future;
    _completer = Completer();

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    Isolate.spawn(_resetIsolate, (params, _sendPort!)).then((value) async {
      receivePort.listen((data) {
        if (data is int) {
          if (data == 0) {
            _completer!.complete();
          } else {
            _completer!.completeError(Exception('Failed to reset LLM, Exception Uncaught'));
          } 
        } else if (data is String && _log != null) {
          _log!(data);
        }
      });

      await _completer!.future;
    });
  }

  Stream<String> prompt(List<ChatMessage> messages) async* {   
    List<ChatMessage> cleanedMessages = messages;
    
    if (_lastMessages != null) {      
      bool same = true;

      int difference = messages.length - _lastMessages!.length;

      if (difference <= 0) {
        same = false;
      } 
      else {
        // Check is all previous messages are the same
        for (var i = 0; i < messages.length - difference; i++) {
          if (_lastMessages![i].contentAsString != messages[i].contentAsString) {
            same = false;
            break;
          }
        }
      }

      // If messages are the same, only send the last message
      if (same) {
        cleanedMessages = messages.sublist(messages.length - difference + 1);
      }
    }

    _lastMessages = messages;
    
    // Ensure initialization is complete
    await _completer?.future;
    _completer = Completer();

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    final isolate = await Isolate.spawn(_promptIsolate, (cleanedMessages, _sendPort!));

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
      final ret1 = lib.maid_llm_model_init(params.get(), Pointer.fromFunction(_logOutput));
      if (ret1 != 0) {
        throw Exception('Failed to initialize model');
      }

      final ret2 = lib.maid_llm_context_init(params.get(), Pointer.fromFunction(_logOutput));
      if (ret2 != 0) {
        throw Exception('Failed to initialize context');
      }

      _sendPort!.send(ret1 + ret2);
    } catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static void _resetIsolate((GptParams, SendPort) args) {
    final (params, sendPort) = args;
    _sendPort = sendPort;

    try {
      final ret = lib.maid_llm_context_init(params.get(), Pointer.fromFunction(_logOutput));

      if (ret != 0) {
        throw Exception('Failed to reset');
      }

      _sendPort!.send(ret);
    } catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static void _promptIsolate((List<ChatMessage>, SendPort) args) {
    final (messages, sendPort) = args;
    _sendPort = sendPort;

    try {
      final ret = lib.maid_llm_prompt(
        messages.length,
        _toNativeChatMessages(messages), 
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

  static Pointer<Pointer<chat_message>> _toNativeChatMessages(
      List<ChatMessage> messages) {
    final chatMessages = calloc<Pointer<chat_message>>(messages.length);

    for (var i = 0; i < messages.length; i++) {
      chatMessages[i] = calloc<chat_message>()
        ..ref.role = _chatMessageToRole(messages[i])
        ..ref.content = messages[i].contentAsString.toNativeUtf8().cast<Char>();
    }

    return chatMessages;
  }

  static int _chatMessageToRole(ChatMessage message) {
    if (message is SystemChatMessage) {
      return chat_role.ROLE_SYSTEM;
    } else if (message is HumanChatMessage) {
      return chat_role.ROLE_USER;
    } else if (message is AIChatMessage) {
      return chat_role.ROLE_ASSISTANT;
    } else {
      throw Exception('Unknown ChatMessage type');
    }
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
