import 'package:flutter/material.dart';

import 'screens/home_screen.dart';

void main() {
  runApp(const SpeechPathologyApp());
}

class SpeechPathologyApp extends StatelessWidget {
  const SpeechPathologyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Stuttering Dysarthria AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.blue,
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}
