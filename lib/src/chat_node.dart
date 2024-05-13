class ChatNode {
  ChatRole role = ChatRole.none;

  String content = '';
  bool finalised = false;

  int _childIndex = 0;
  List<ChatNode> children = [];

  ChatNode({
    required this.role,
    this.content = "",
    this.finalised = false,
    List<ChatNode>? children,
  }) : children = children ?? [];

  ChatNode.fromMap(Map<String, dynamic> map) {
    role = ChatRole.values.firstWhere((role) => role.name == map['role']);
    content = map['content'];
    finalised = true;
    _childIndex = map['child'] ?? 0;
    children = (map['children'] as List).map((child) => ChatNode.fromMap(child)).toList();
  }

  Map<String, dynamic> toMap() {
    return {
      'role': role.name,
      'content': content,
      'child': _childIndex,
      'children': children.map((child) => child.toMap()).toList(),
    };
  }

  void next() {
    if (_childIndex < children.length - 1) {
      _childIndex++;
    }
  }

  void last() {
    if (_childIndex > 0) {
      _childIndex--;
    }
  }
}

enum ChatRole {
  none,
  system,
  user,
  assistant
}