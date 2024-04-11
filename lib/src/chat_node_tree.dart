import 'dart:async';
import 'dart:collection';

import 'package:flutter/material.dart';
import 'package:maid_llm/src/chat_node.dart';

class ChatNodeTree {
  String buffer = "";

  ChatNode root = ChatNode(key: UniqueKey(), role: ChatRole.system);

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
    final node = ChatNode(key: key, content: content, role: role);

    var found = find(key);
    if (found != null) {
      found.content = content;
    } 
    else {
      tail.children.add(node);
      tail.currentChild = key;
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
      if (parent.currentChild == null) {
        parent.currentChild = key;
      } 
      else {
        var currentChildIndex = parent.children.indexWhere(
          (element) => element.key == parent.currentChild
        );

        if (currentChildIndex < parent.children.length - 1) {
          parent.currentChild = parent.children[currentChildIndex + 1].key;
        }
      }
    }
  }

  void last(Key key) {
    var parent = parentOf(key);
    if (parent != null) {
      if (parent.currentChild == null) {
        parent.currentChild = key;
      } 
      else {
        var currentChildIndex = parent.children.indexWhere(
          (element) => element.key == parent.currentChild
        );

        if (currentChildIndex > 0) {
          parent.currentChild = parent.children[currentChildIndex - 1].key;
        }
      }
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

  StreamController<String> getMessageStream(Key key) {
    return find(key)?.messageController ??
        StreamController<String>.broadcast();
  }

  Map<Key, ChatRole> getHistory() {
    final Map<Key, ChatRole> history = {};
    var current = root;

    while (current.currentChild != null) {
      current = find(current.currentChild!)!;
      history[current.key] = current.role;
    }

    return history;
  }

  List<Map<String, dynamic>> getMessages() {
    final List<Map<String, dynamic>> messages = [];
    var current = root;

    while (current.currentChild != null) {
      current = find(current.currentChild!)!;
      messages.add({"role": current.role.name, "content": current.content});
    }

    if (messages.isNotEmpty) {
      messages.remove(messages.last); //remove last message
    }

    return messages;
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