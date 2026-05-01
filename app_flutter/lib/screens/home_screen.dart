import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import '../services/prediction_api.dart';

enum AppState {
  idle,
  recording,
  recorded,
  predicting,
  done,
  error,
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  final PredictionApi _api = PredictionApi();

  AppState _state = AppState.idle;
  File? _audioFile;
  PredictionResult? _result;
  String? _error;

  Future<void> _recordThreeSeconds() async {
    try {
      final hasPermission = await _recorder.hasPermission();

      if (!hasPermission) {
        setState(() {
          _state = AppState.error;
          _error = 'Microphone permission denied.';
        });
        return;
      }

      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/speech_sample.wav';

      setState(() {
        _state = AppState.recording;
        _result = null;
        _error = null;
        _audioFile = null;
      });

      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
        ),
        path: path,
      );

      await Future.delayed(const Duration(seconds: 3));

      final recordedPath = await _recorder.stop();

      setState(() {
        _audioFile = File(recordedPath ?? path);
        _state = AppState.recorded;
      });
    } catch (e) {
      setState(() {
        _state = AppState.error;
        _error = e.toString();
      });
    }
  }

  Future<void> _analyzeSpeech() async {
    if (_audioFile == null) return;

    try {
      setState(() {
        _state = AppState.predicting;
        _error = null;
      });

      final result = await _api.predictAudio(_audioFile!);

      setState(() {
        _result = result;
        _state = AppState.done;
      });
    } catch (e) {
      setState(() {
        _state = AppState.error;
        _error = e.toString();
      });
    }
  }

  String _statusText() {
    switch (_state) {
      case AppState.idle:
        return 'Record a 3-second speech sample.';
      case AppState.recording:
        return 'Recording... speak naturally.';
      case AppState.recorded:
        return 'Recording ready. Tap Analyze.';
      case AppState.predicting:
        return 'Analyzing speech...';
      case AppState.done:
        return 'Analysis complete.';
      case AppState.error:
        return 'Something went wrong.';
    }
  }

  Color _predictionColor() {
    if (_result?.prediction == 'stutter') return Colors.orange;
    if (_result?.prediction == 'fluent') return Colors.green;
    return Colors.blueGrey;
  }

  String _percent(double value) {
    return '${(value * 100).toStringAsFixed(1)}%';
  }

  @override
  void dispose() {
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isBusy =
        _state == AppState.recording || _state == AppState.predicting;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Speech Pathology AI'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(18),
          children: [
            Card(
              elevation: 2,
              child: Padding(
                padding: const EdgeInsets.all(22),
                child: Column(
                  children: [
                    Icon(
                      Icons.mic_rounded,
                      size: 76,
                      color: _state == AppState.recording
                          ? Colors.red
                          : Theme.of(context).colorScheme.primary,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      _statusText(),
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Speak naturally for 3 seconds. Use a quiet place for better results.',
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 18),
            ElevatedButton.icon(
              onPressed: isBusy ? null : _recordThreeSeconds,
              icon: const Icon(Icons.fiber_manual_record_rounded),
              label: const Text('Record 3 Seconds'),
            ),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: _audioFile == null || isBusy ? null : _analyzeSpeech,
              icon: const Icon(Icons.analytics_rounded),
              label: const Text('Analyze Speech'),
            ),
            const SizedBox(height: 24),
            if (_state == AppState.predicting)
              const Center(child: CircularProgressIndicator()),
            if (_result != null)
              Card(
                color: _predictionColor().withOpacity(0.12),
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Prediction: ${_result!.prediction.toUpperCase()}',
                        style: Theme.of(context).textTheme.headlineSmall,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Confidence: ${_percent(_result!.confidence)}',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Fluent probability: ${_percent((_result!.probabilities['fluent'] ?? 0).toDouble())}',
                      ),
                      Text(
                        'Stutter probability: ${_percent((_result!.probabilities['stutter'] ?? 0).toDouble())}',
                      ),
                      const SizedBox(height: 12),
                      Text('Model: ${_result!.modelName}'),
                      Text('Version: ${_result!.modelVersion}'),
                      Text('Device: ${_result!.device}'),
                      Text(
                        'Inference time: ${_result!.inferenceSeconds.toStringAsFixed(2)} sec',
                      ),
                      const SizedBox(height: 14),
                      Text(
                        _result!.warning,
                        style: const TextStyle(
                          fontSize: 12,
                          color: Colors.black54,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            if (_error != null)
              Card(
                color: Colors.red.withOpacity(0.12),
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Text(_error!),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
