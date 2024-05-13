import 'dart:collection';

import 'package:flutter/material.dart';
import 'package:maid_llm/src/chat_node.dart';

class ChatNodeTree {
  String buffer = "";

  ChatNode root = ChatNode(key: UniqueKey(), role: ChatRole.system, finalised: true);

  ChatNode get tail {
    final Queue<ChatNode> queue = Queue.from([root]);

    while (queue.isNotEmpty) {
      final current = queue.removeFirst();

      if (current.children.isEmpty || current.currentChild == null) {
        return current;
      } 
      else {
        for (var child in current.children) {
          if (child.key == current.currentChild) {
            queue.add(child);
          }
        }
      }
    }

    return root;
  }

  void add(
    Key key, {
    String content = "",
    ChatRole role = ChatRole.user,
  }) {
    final node = ChatNode(key: key, content: content, role: role, finalised: content.isNotEmpty);

    var found = find(key);
    if (found != null) {
      found.content = content;
    } 
    else {
      tail.children.add(node);
      tail.currentChild = key;
    }
  }

  void addNode(ChatNode node) {
    var found = find(node.key);
    if (found != null) {
      found.content = node.content;
    } 
    else {
      tail.children.add(node);
      tail.currentChild = node.key;
    }
  }

  void remove(Key key) {
    var parent = parentOf(key);
    if (parent != null) {
      parent.children.removeWhere((element) => element.key == key);
    }
  }

  void next(Key key) {
    var parent = parentOf(key);
    if (parent != null) {
      parent.next();
    }
  }

  void last(Key key) {
    var parent = parentOf(key);
    if (parent != null) {
      parent.last();
    }
  }

  ChatNode? find(Key targetKey) {
    final Queue<ChatNode> queue = Queue.from([root]);

    while (queue.isNotEmpty) {
      final current = queue.removeFirst();

      if (current.key == targetKey) {
        return current;
      }

      for (var child in current.children) {
        queue.add(child);
      }
    }

    return null;
  }

  ChatNode? parentOf(Key targetKey) {
    final Queue<ChatNode> queue = Queue.from([root]);

    while (queue.isNotEmpty) {
      final current = queue.removeFirst();

      for (var child in current.children) {
        if (child.key == targetKey) {
          return current;
        }
        queue.add(child);
      }
    }

    return null;
  }

  String messageOf(Key key) {
    return find(key)?.content ?? "";
  }

  int indexOf(Key key) {
    var parent = parentOf(key);
    if (parent != null) {
      return parent.children.indexWhere((element) => element.key == key);
    } else {
      return 0;
    }
  }

  int siblingCountOf(Key key) {
    var parent = parentOf(key);
    if (parent != null) {
      return parent.children.length;
    } else {
      return 0;
    }
  }

  List<ChatNode> getChat() {
    final List<ChatNode> chat = [];
    var current = root;

    while (current.currentChild != null) {
      current = find(current.currentChild!)!;
      if (current.content.isNotEmpty) {
        chat.add(current);
      }
    }

    return chat;
  }
}