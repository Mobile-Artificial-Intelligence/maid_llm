import 'package:flutter_test/flutter_test.dart';
import 'package:maid_llm/src/chat_node.dart';

void main() {
  group('ChatNode', () {
    test('should instantiate with default values correctly', () {
      final node = ChatNode(role: ChatRole.user);

      expect(node.hash, isNotEmpty);
      expect(node.role, equals(ChatRole.user));
      expect(node.content, isEmpty);
      expect(node.finalised, isFalse);
      expect(node.children, isEmpty);
    });

    test('should serialize and deserialize correctly', () {
      final node = ChatNode(role: ChatRole.user, content: 'Hello, world!', finalised: true);
      final map = node.toMap();

      final newNode = ChatNode.fromMap(map);
      expect(newNode.hash, equals(node.hash));
      expect(newNode.role, equals(node.role));
      expect(newNode.content, equals(node.content));
      expect(newNode.finalised, equals(node.finalised));
    });

    test('should manage children correctly', () {
      final parent = ChatNode(role: ChatRole.system);
      final child = ChatNode(role: ChatRole.user);

      parent.children.add(child);
      expect(parent.children, contains(child));
    });
  });
}
