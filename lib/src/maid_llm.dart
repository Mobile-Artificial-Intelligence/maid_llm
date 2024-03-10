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

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    _completer = Completer();

    Isolate.spawn(_initIsolate, (params, _sendPort!)).then((value) async {
      receivePort.listen((message) {
        if (message is int) {
          if (message == 0) {
            _completer!.complete();
          } else {
            _completer!.completeError(Exception('Failed to initialize LLM'));
          }
        }
      });

      await _completer!.future;
    });
  }

  Stream<String> prompt(List<ChatMessage> messages) async* {
    await _completer!.future;
    _completer = Completer();

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    final isolate = await Isolate.spawn(_promptIsolate, (messages, _sendPort!));

    await for (var data in receivePort) {
      if (data is (String, bool)) {
        final (message, done) = data;

        if (done) {
          receivePort.close();
          isolate.kill();
          _completer!.complete();
          return;
        }

        yield message;
      } else if (data is String) {
        _log?.call(data);
      }
    }
  }

  static void _initIsolate((GptParams, SendPort) args) {
    final (params, sendPort) = args;
    _sendPort = sendPort;

    final ret =
        lib.maid_llm_init(params.get(), Pointer.fromFunction(_logOutput));

    _sendPort!.send(ret);
  }

  static void _promptIsolate((List<ChatMessage>, SendPort) args) {
    final (messages, sendPort) = args;
    _sendPort = sendPort;

    try {
      final ret = lib.maid_llm_prompt(
        messages.length,
        _toNativeChatMessages(messages), 
        Pointer.fromFunction(_output)
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
    } catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static void _logOutput(Pointer<Char> message) {
    _sendPort!.send(message.cast<Utf8>().toDartString());
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
