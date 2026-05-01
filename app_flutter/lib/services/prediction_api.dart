import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../config.dart';

class PredictionResult {
  final String prediction;
  final double confidence;
  final Map<String, dynamic> probabilities;
  final String modelName;
  final String modelVersion;
  final double inferenceSeconds;
  final String device;
  final String warning;

  PredictionResult({
    required this.prediction,
    required this.confidence,
    required this.probabilities,
    required this.modelName,
    required this.modelVersion,
    required this.inferenceSeconds,
    required this.device,
    required this.warning,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      prediction: json['prediction'] ?? 'unknown',
      confidence: (json['confidence'] ?? 0).toDouble(),
      probabilities: Map<String, dynamic>.from(json['probabilities'] ?? {}),
      modelName: json['model_name'] ?? 'unknown',
      modelVersion: json['model_version'] ?? 'unknown',
      inferenceSeconds: (json['inference_seconds'] ?? 0).toDouble(),
      device: json['device'] ?? 'unknown',
      warning: json['warning'] ?? '',
    );
  }
}

class PredictionApi {
  Future<PredictionResult> predictAudio(File audioFile) async {
    final uri = Uri.parse('${AppConfig.apiBaseUrl}/v1/predict');

    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      await http.MultipartFile.fromPath(
        'file',
        audioFile.path,
        filename: 'speech_sample.wav',
      ),
    );

    final response = await request.send();
    final body = await response.stream.bytesToString();

    if (response.statusCode != 200) {
      throw Exception('Prediction failed: ${response.statusCode}\n$body');
    }

    return PredictionResult.fromJson(jsonDecode(body));
  }
}
