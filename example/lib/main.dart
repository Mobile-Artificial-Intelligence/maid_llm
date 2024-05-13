import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

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
  List<ChatNode> messages = [];
  String modelPath = "";

  Future<String> loadModel() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        dialogTitle: "Load Model File",
        type: FileType.any,
        allowMultiple: false,
        allowCompression: false
      );

      File file;
      if (result != null && result.files.isNotEmpty) {
        file = File(result.files.single.path!);
      } else {
        throw Exception("File is null");
      }

      setState(() {
        modelPath = file.path;
      });
    } catch (e) {
      return e.toString();
    }

    return "Model Successfully Loaded";
  }

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    const textStyle = TextStyle(fontSize: 25);
    const spacerSmall = SizedBox(height: 10);
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text(modelPath.isEmpty ? "No Model Loaded" : modelPath),
          actions: [
            IconButton(
              icon: const Icon(Icons.file_download),
              onPressed: () {
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                  content: FutureBuilder<String>(
                    future: loadModel(), 
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.done) {
                        return Text(snapshot.data!);
                      } else {
                        return const CircularProgressIndicator();
                      }
                    }
                  ),
                ));
              },
            ),
          ],
        ),
        body: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(10),
            child: const Column(
              children: [
                Text(
                  'This calls a native function through FFI that is shipped as source in the package. '
                  'The native code is built as part of the Flutter Runner build.',
                  style: textStyle,
                  textAlign: TextAlign.center,
                ),
                spacerSmall,
              ],
            ),
          ),
        ),
      ),
    );
  }
}
