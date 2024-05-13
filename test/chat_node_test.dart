import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:maid_llm/src/chat_node.dart';

void main() {
  group('ChatNode', () {
    test('should instantiate with default values correctly', () {
      final key = UniqueKey();
      final node = ChatNode(key: key, role: ChatRole.user);

      expect(node.key, equals(key));
      expect(node.role, equals(ChatRole.user));
      expect(node.content, isEmpty);
      expect(node.finalised, isFalse);
      expect(node.children, isEmpty);
    });

    test('should serialize and deserialize correctly', () {
      final key = UniqueKey();
      final node = ChatNode(key: key, role: ChatRole.user, content: 'Hello, world!', finalised: true);
      final map = node.toMap();

      final newNode = ChatNode.fromMap(map);
      expect(newNode.key, equals(node.key));
      expect(newNode.role, equals(node.role));
      expect(newNode.content, equals(node.content));
      expect(newNode.finalised, equals(node.finalised));
    });

    test('should manage children correctly', () {
      final parentKey = UniqueKey();
      final childKey = UniqueKey();
      final parent = ChatNode(key: parentKey, role: ChatRole.system);
      final child = ChatNode(key: childKey, role: ChatRole.user);

      parent.children.add(child);
      expect(parent.children, contains(child));
    });
  });
}
