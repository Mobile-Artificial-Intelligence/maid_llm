import 'dart:ffi';
import 'dart:math';

import 'package:ffi/ffi.dart';

import 'bindings.dart';

class ChatMessage {
  final String role;
  final String content;

  ChatMessage({
    required this.role,
    required this.content,
  });

  Pointer<llama_chat_message> toNative() {
    final message = calloc<llama_chat_message>();
    message.ref.role = role.toNativeUtf8().cast<Char>();
    message.ref.content = content.toNativeUtf8().cast<Char>();

    return message;
  }
}