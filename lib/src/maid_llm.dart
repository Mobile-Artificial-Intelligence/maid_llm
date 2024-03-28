import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:langchain/langchain.dart';
import 'gpt_params.dart';

import 'maid_llm_bindings.dart';

class MaidLLM {
  static Completer? _completer;
  static SendPort? _sendPort;
  static maid_llm? _lib;
  static void Function(String)? _log;

  /// Getter for the maid_llm library.
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

  static void loadModel(GptParams params) {

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    _completer = Completer();

    Isolate.spawn(_loadModelIsolate, (params, _sendPort!)).then((value) async {
      receivePort.listen((data) {
        if (data is int) {
          if (data == 0) {
            _completer!.complete();
          } else {
            _completer!.completeError(Exception('Failed to initialize LLM'));
          } 
        } else if (data is String && _log != null) {
          _log!(data);
        }
      });

      await _completer!.future;
    });
  }

  static Stream<String> prompt(GptParams params, List<ChatMessage> messages) async* {
    await _completer!.future;
    _completer = Completer();

    final receivePort = ReceivePort();
    _sendPort = receivePort.sendPort;

    final isolate = await Isolate.spawn(_promptIsolate, (messages, params, _sendPort!));

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
      } else if (data is String && _log != null) {
        _log!(data);
      }
    }
  }

  static void _loadModelIsolate((GptParams, SendPort) args) {
    final (params, sendPort) = args;
    _sendPort = sendPort;

    final ret = lib.maid_llm_load_model(params.get());

    _sendPort!.send(ret);
  }

  static void _promptIsolate((List<ChatMessage>, GptParams, SendPort) args) {
    final (messages, params, sendPort) = args;
    _sendPort = sendPort;

    try {
      final outputs = calloc<dart_outputs>()
        ..ref.log = Pointer.fromFunction(_logOutput)
        ..ref.chat = Pointer.fromFunction(_output);

      final ret = lib.maid_llm_prompt(
        params.get(),
        _toNativeChat(messages),
        outputs
      );

      if (ret != 0) {
        throw Exception('Failed to prompt');
      }
    } catch (e) {
      _sendPort!.send(e.toString());
    }
  }

  static Pointer<maid_llm_chat> _toNativeChat(
      List<ChatMessage> messages) {
    final chatMessages = calloc<llama_chat_message>(messages.length);

    int length = 0;
    for (var i = 0; i < messages.length; i++) {
      final content = messages[i].contentAsString;

      final message = calloc<llama_chat_message>()
        ..ref.role = _chatMessageToRole(messages[i]).toNativeUtf8().cast<Char>()
        ..ref.content = content.toNativeUtf8().cast<Char>();

      chatMessages[i] = message.ref;

      length += content.length;
    }

    final chat = calloc<maid_llm_chat>()
      ..ref.messages = chatMessages
      ..ref.n_messages = messages.length
      ..ref.length = length;

    return chat;
  }

  static String _chatMessageToRole(ChatMessage message) {
    if (message is SystemChatMessage) {
      return "system";
    } else if (message is HumanChatMessage) {
      return "user";
    } else if (message is AIChatMessage) {
      return "assistant";
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
    final logMessage = message.cast<Utf8>().toDartString();

    _sendPort!.send(logMessage);
  }

  static Future<void> stop() async {
    lib.maid_llm_stop();
    await _completer!.future;
    return;
  }

  void clear() {
    lib.maid_llm_cleanup();
  }
}
