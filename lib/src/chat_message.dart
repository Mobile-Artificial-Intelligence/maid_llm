part of '../maid_llm.dart';

class ChatMessage {
  final String role;
  final String content;

  ChatMessage({
    required this.role,
    required this.content,
  });

  ChatMessage.fromNative(Pointer<llama_chat_message> message)
      : role = message.ref.role.toString(),
        content = message.ref.content.toString();

  Pointer<llama_chat_message> toNative() {
    final message = calloc<llama_chat_message>();
    message.ref.role = role.toNativeUtf8().cast<Char>();
    message.ref.content = content.toNativeUtf8().cast<Char>();

    return message;
  }
}