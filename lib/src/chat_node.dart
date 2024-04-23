import 'dart:async';

import 'package:flutter/material.dart';

class ChatNode {
  final StreamController<String> messageController = StreamController<String>.broadcast();

  late Key key;
  late ChatRole role;

  String content = '';

  Key? currentChild;
  List<ChatNode> children = [];

  ChatNode({
    required this.key,
    required this.role,
    this.content = "",
    List<ChatNode>? children,
  }) : children = children ?? [];

  ChatNode.fromMap(Map<String, dynamic> map) {
    key = Key(map['key']);
    role = ChatRole.values[map['role']];
    content = map['content'];
    currentChild = map['currentChild'] != null ? Key(map['currentChild']) : null;
    children = (map['children'] as List).map((child) => ChatNode.fromMap(child)).toList();
  }

  Map<String, dynamic> toMap() {
    return {
      'key': _keyToString(key),
      'role': role.index,
      'content': content,
      'currentChild': currentChild != null ? _keyToString(currentChild!) : null,
      'children': children.map((child) => child.toMap()).toList(),
    };
  }

  static String _keyToString(Key key) {
    String keyString = key.toString();
    
    keyString = keyString.replaceAll('\'', '');
    keyString = keyString.replaceAll('<', '');
    keyString = keyString.replaceAll('>', '');
    keyString = keyString.replaceAll('[', '');
    keyString = keyString.replaceAll(']', '');

    return keyString;
  }
}

enum ChatRole {
  system,
  user,
  assistant
}