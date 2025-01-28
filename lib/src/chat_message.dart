part of '../lcpp.dart';

class ChatMessage {
  final String role;
  final String content;

  ChatMessage({
    required this.role,
    required this.content,
  });

  ChatMessage.fromNative(ffi.Pointer<llama_chat_message> message)
      : role = message.ref.role.toString(),
        content = message.ref.content.toString();

  ffi.Pointer<llama_chat_message> toNative() {
    final message = calloc<llama_chat_message>();
    message.ref.role = role.toNativeUtf8().cast<ffi.Char>();
    message.ref.content = content.toNativeUtf8().cast<ffi.Char>();

    return message;
  }
}