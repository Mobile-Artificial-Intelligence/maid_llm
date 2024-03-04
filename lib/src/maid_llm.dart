import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:langchain/langchain.dart';
import 'gpt_params.dart';

import 'bindings.dart';

class MaidLLM {
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
    lib.maid_llm_init(params.get(), Pointer.fromFunction(_logOutput));
  }

  Stream<String> prompt(List<ChatMessage> messages) async* {
    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    final isolate = await Isolate.spawn(_promptIsolate, (messages, _sendPort!));

    await for (var data in receivePort) {
      if (data is (String, bool)) {
        final (message, done) = data;

        if (done) {
          receivePort.close();
          isolate.kill();
          return;
        }

        yield message;
      }
    }
  }

  static void _promptIsolate((List<ChatMessage>, SendPort) args) {
    final (messages, sendPort) = args;
    _sendPort = sendPort;
    lib.maid_llm_prompt(
        _toNativeChatMessages(messages), Pointer.fromFunction(_output));
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
      print(e);
    }
  }

  static void _logOutput(Pointer<Char> message) {
    if (_log != null) {
      _log!(message.cast<Utf8>().toDartString());
    }
  }

  void stop() {
    lib.maid_llm_stop();
  }

  void clear() {
    lib.maid_llm_cleanup();
  }
}
