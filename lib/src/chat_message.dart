part of '../lcpp.dart';

class ChatMessage {
  final String role;
  final String content;

  ChatMessage({
    required this.role,
    required this.content,
  });

  ChatMessage.fromNative(llama_chat_message message)
      : role = message.role.cast<Utf8>().toDartString(),
        content = message.content.cast<Utf8>().toDartString();

  llama_chat_message toNative() {
    final message = calloc<llama_chat_message>();
    message.ref.role = role.toNativeUtf8().cast<ffi.Char>();
    message.ref.content = content.toNativeUtf8().cast<ffi.Char>();

    return message.ref;
  }
}

extension ChatMessages on List<ChatMessage> {
  ffi.Pointer<llama_chat_message> toNative() {
    final messages = calloc<llama_chat_message>(length);

    for (var i = 0; i < length; i++) {
      messages[i] = this[i].toNative();
    }

    return messages;
  }
}