import 'package:flutter/material.dart';
import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:maid_llm/maid_llm.dart';

void main() {
  runApp(const MaidLlmApp());
}

class MaidLlmApp extends StatefulWidget {
  const MaidLlmApp({super.key});

  @override
  State<MaidLlmApp> createState() => _MaidLlmAppState();
}

class _MaidLlmAppState extends State<MaidLlmApp> {
  final TextEditingController _controller = TextEditingController();
  final List<ChatMessage> _messages = [];
  String? _model;

  void _loadModel() async {
    final result = await FilePicker.platform.pickFiles(
      dialogTitle: "Load Model File",
      type: FileType.any,
      allowMultiple: false,
      allowCompression: false
    );

    if (result != null && result.files.isNotEmpty) {
      setState(() {
        _model = result.files.single.path!;
      });
    }
  }

  void _onSubmitted(String value) async {
    if (_model == null) {
      return;
    }

    setState(() {
      _messages.add(ChatMessage(role: 'user', content: value));
      _controller.clear();
    });

    GptParams gptParams = GptParams();
    gptParams.model = _model!;

    Stream<String> stream = MaidLLM(gptParams).prompt(_messages, "");

    setState(() {
      _messages.add(ChatMessage(role: 'assistant', content: ""));
    });

    stream.listen((message) {
      setState(() {
        final newContent = _messages.last.content + message;
        final newLastMessage = ChatMessage(role: 'assistant', content: newContent);
        _messages[_messages.length - 1] = newLastMessage;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: buildHome()
    );
  }

  Widget buildHome() {
    return Scaffold(
      appBar: buildAppBar(),
      body: buildBody(),
    );
  }

  PreferredSizeWidget buildAppBar() {
    return AppBar(
      title: Text(_model ?? 'No model loaded'),
      leading: IconButton(
        icon: const Icon(Icons.folder_open),
        onPressed: _loadModel,
      ),
    );
  }

  Widget buildBody() {
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            itemCount: _messages.length,
            itemBuilder: (context, index) {
              final message = _messages[index];
              return ListTile(
                title: Text(message.role),
                subtitle: Text(message.content),
              );
            },
          ),
        ),
        buildInputField(),
      ],
    );
  }

  Widget buildInputField() {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              onSubmitted: _onSubmitted,
              decoration: const InputDecoration(
                labelText: 'Enter your message',
                border: OutlineInputBorder(),
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.send),
            onPressed: () {
              _onSubmitted(_controller.text);
            },
          ),
        ],
      ),
    );
  }
}