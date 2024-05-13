import 'dart:collection';

import 'package:flutter/material.dart';
import 'package:maid_llm/src/chat_node.dart';

class ChatNodeTree {
  String buffer = "";

  ChatNode root = ChatNode(role: ChatRole.system, finalised: true);

  ChatNode get tail {
    return root.tail;
  }

  void add(
    String hash, {
    String content = "",
    ChatRole role = ChatRole.user,
  }) {
    final node = ChatNode(content: content, role: role, finalised: content.isNotEmpty);

    var found = find(hash);
    if (found != null) {
      found.content = content;
    } 
    else {
      tail.child = node;
    }
  }

  void addNode(ChatNode node) {
    var found = find(node.hash);
    if (found != null) {
      found.content = node.content;
    } 
    else {
      tail.child = node;
    }
  }

  ChatNode? find(String hash) {
    final Queue<ChatNode> queue = Queue.from([root]);

    while (queue.isNotEmpty) {
      final current = queue.removeFirst();

      if (current.hash == hash) {
        return current;
      }

      for (var child in current.children) {
        queue.add(child);
      }
    }

    return null;
  }

  ChatNode? parentOf(String hash) {
    final Queue<ChatNode> queue = Queue.from([root]);

    while (queue.isNotEmpty) {
      final current = queue.removeFirst();

      for (var child in current.children) {
        if (child.hash == hash) {
          return current;
        }
        queue.add(child);
      }
    }

    return null;
  }

  List<ChatNode> getChat() {
    final List<ChatNode> chat = [];
    var current = root;

    while (current.child != null) {
      current = current.child!;
      if (current.content.isNotEmpty) {
        chat.add(current);
      }
    }

    return chat;
  }
}