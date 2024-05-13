import 'package:crypto/crypto.dart';

class ChatNode {
  ChatRole role = ChatRole.none;

  String content = '';
  bool finalised = false;

  List<ChatNode> _children = [];

  final DateTime created;

  set child(ChatNode? value) {
    if (value != null) {
      _children.insert(0, value);
    }
  }

  String get hash {
    final childHashes = _children.map((child) => child.hash).join();
    final timeString = created.toIso8601String();
    final strings = "$role$content$childHashes$timeString";
    final hash = sha256.convert(strings.codeUnits);
    return hash.toString();
  }

  ChatNode? get child {
    return _children.firstOrNull;
  }

  ChatNode get tail {
    for (final node in _children) {
      if (node.child == null) {
        return node;
      } 
      else {
        return node.tail;
      }
    }

    return this;
  }

  int get childCount {
    return _children.length;
  }

  ChatNode({
    required this.role,
    this.content = "",
    this.finalised = false,
    List<ChatNode>? children
  }) : created = DateTime.now() {
    children = children ?? [];
  }

  ChatNode.fromMap(Map<String, dynamic> map) : created = DateTime.now(), assert(map['role'] != null) {
    role = ChatRole.values.firstWhere((role) => role.name == map['role']);
    content = map['content'];
    finalised = true;
    _children = (map['children'] as List).map((child) => ChatNode.fromMap(child)).toList();
  }

  Map<String, dynamic> toMap() {
    return {
      'role': role.name,
      'content': content,
      'children': _children.map((child) => child.toMap()).toList()
    };
  }

  ChatNode? find(String value) {
    if (hash == value) {
      return this;
    }

    for (var child in _children) {
      final found = child.find(value);
      if (found != null) {
        return found;
      }
    }

    return null;
  }

  void next() {
    // First child is moved to the end and the rest are pushed forward
    if (_children.isNotEmpty) {
      final first = _children.removeAt(0);
      _children.add(first);
    }
  }

  void last() {
    // Last child is moved to the front and the rest are pushed back
    if (_children.isNotEmpty) {
      final last = _children.removeLast();
      _children.insert(0, last);
    }
  }
}

enum ChatRole {
  none,
  system,
  user,
  assistant
}