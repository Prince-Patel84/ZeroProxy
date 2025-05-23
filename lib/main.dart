import 'dart:convert';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'dart:io' show File;
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

import 'Recognizer.dart';

import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image_picker/image_picker.dart';

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class FacePainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final bool isFrontCamera;

  FacePainter(this.faces, this.imageSize, this.isFrontCamera);

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.red;

    final landmarkPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.green
      ..strokeWidth = 2.0;

    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;

    for (Face face in faces) {
      final rect = face.boundingBox;

      double left = rect.left * scaleX;
      double top = rect.top * scaleY;
      double right = rect.right * scaleX;
      double bottom = rect.bottom * scaleY;

      if (isFrontCamera) {
        double tempLeft = left;
        left = size.width - right;
        right = size.width - tempLeft;
      }

      // Draw bounding box
      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), boxPaint);

      // Draw landmarks
      final landmarkTypes = [
        FaceLandmarkType.leftEye,
        FaceLandmarkType.rightEye,
        FaceLandmarkType.noseBase,
        FaceLandmarkType.leftEar,
        FaceLandmarkType.rightEar,
        FaceLandmarkType.bottomMouth,
        FaceLandmarkType.leftMouth,
        FaceLandmarkType.rightMouth,
      ];

      for (var type in landmarkTypes) {
        final landmark = face.landmarks[type];
        if (landmark != null) {
          double x = landmark.position.x * scaleX;
          double y = landmark.position.y * scaleY;

          if (isFrontCamera) {
            x = size.width - x;
          }

          canvas.drawCircle(Offset(x, y), 4.0, landmarkPaint);
        }
      }
    }
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) => true;
}

class CroppedFace {
  final img.Image image;
  final Uint8List bytes;

  CroppedFace(this.image, this.bytes);
}

class SuccessScreen extends StatelessWidget {
  const SuccessScreen({super.key});

  @override
  Widget build(BuildContext context) {
    Future.delayed(Duration(seconds: 2), () {
      Navigator.pop(context); // Go back to the previous page
    });

    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.check_circle, color: Colors.green, size: 100),
            SizedBox(height: 20),
            Text(
              'Student Registered Successfully!',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }
}


class _MyHomePageState extends State<MyHomePage> {
  //Variables
  late ImagePicker imagePicker;
  late FaceDetector faceDetector;
  File? _image;
  Size _imageSize = Size.zero;

  List<Face> _faces = [];
  List<CroppedFace> _croppedFaces = [];
  List<List<double>> _faceEmbeddings = [];

  late TextEditingController studentIdController;
  late TextEditingController passwordController;

  bool _uploadSuccess = false;
  bool _isProcessing = false;
  final bool _isFrontCamera = false;

  @override
  void initState() {
    super.initState();
    imagePicker = ImagePicker();
    studentIdController = TextEditingController();
    passwordController = TextEditingController();

    final options = FaceDetectorOptions(
      enableLandmarks: true,
      enableContours: true,
      performanceMode: FaceDetectorMode.accurate,
    );
    faceDetector = FaceDetector(options: options);
  }

  @override
  void dispose() {
    studentIdController.dispose();
    passwordController.dispose();
    faceDetector.close();
    super.dispose();
  }

  Future<List<CroppedFace>> cropFaces(File imageFile, List<Face> faces) async {
    final bytes = await imageFile.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final originalImage = frame.image;

    List<CroppedFace> cropped = [];

    for (var face in faces) {
      final rect = face.boundingBox;

      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder);
      final paint = Paint();

      canvas.drawImageRect(
        originalImage,
        Rect.fromLTWH(
          rect.left.clamp(0, originalImage.width.toDouble()),
          rect.top.clamp(0, originalImage.height.toDouble()),
          rect.width.clamp(1, originalImage.width - rect.left),
          rect.height.clamp(1, originalImage.height - rect.top),
        ),
        Rect.fromLTWH(0, 0, rect.width, rect.height),
        paint,
      );

      final picture = recorder.endRecording();
      final croppedUiImage = await picture.toImage(rect.width.toInt(), rect.height.toInt());

      final byteData = await croppedUiImage.toByteData(format: ui.ImageByteFormat.png);
      final Uint8List imgBytes = byteData!.buffer.asUint8List();

      final decoded = img.decodeImage(imgBytes);
      if (decoded != null) {
        cropped.add(CroppedFace(decoded, imgBytes));
      }
    }

    return cropped;
  }


  Future<Size> _getImageSize(File imageFile) async {
    final decodedImage = await decodeImageFromList(await imageFile.readAsBytes());
    return Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
  }

  Future<File?> captureImages() async {
    final XFile? photo = await imagePicker.pickImage(source: ImageSource.camera);
    if (photo != null) {
      return File(photo.path);
    }
    return null;
  }

  Future<void> captureAndDetect() async {
    try {
      setState(() {
        _isProcessing = true;  // Start loading
      });

      final File? capturedImage = await captureImages();
      if (capturedImage == null) {
        setState(() {
          _isProcessing = false;
        });
        return;
      }

      final inputImage = InputImage.fromFilePath(capturedImage.path);
      final faces = await faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        print("No faces detected.");
        setState(() {
          _isProcessing = false;
        });
        return;
      }

      final decodedImage = await decodeImageFromList(await capturedImage.readAsBytes());

      final List<CroppedFace> cropped = await cropFaces(capturedImage, faces);

      final recognizer = Recognizer();
      await recognizer.loadModel();

      final List<List<double>> embeddings = [];

      for (final faceImg in cropped) {
        final input = recognizer.imageToArray(faceImg.image);
        final output = List.filled(192, 0).reshape([1, 192]);

        recognizer.interpreter.run(input, output);

        embeddings.add(List<double>.from(output[0]));
      }

      recognizer.close();

      setState(() {
        _image = capturedImage;
        _faces = faces;
        _imageSize = Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
        _croppedFaces = cropped;
        _faceEmbeddings = embeddings;
        _isProcessing = false;  // Stop loading
      });
    } catch (e, st) {
      print("Error in captureAndDetect: $e\n$st");
      setState(() {
        _isProcessing = false;
      });
    }
  }

  // New function to send data to server
  Future<void> sendDataToServer({
    required String studentId,
    required String password,
    required File imageFile,
    required List<List<double>> faceEmbeddings,
  }) async {
    final url = Uri.parse('http://192.168.48.84:3000/upload-image'); // update IP

    final request = http.MultipartRequest('POST', url)
      ..fields['student_id'] = studentId
      ..fields['password'] = password
      ..fields['embeddings'] = jsonEncode(faceEmbeddings)
      ..files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    try {
      final response = await request.send();
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        print("Upload successful");
        setState(() {
          _uploadSuccess = true;
        });

        if (!mounted) return;

        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => const SuccessScreen()),
        );

        setState(() {
          _uploadSuccess = false;
          studentIdController.clear();
          passwordController.clear();
          _image = null;
          _faces.clear();
          _croppedFaces.clear();
          _faceEmbeddings.clear();
        });

      } else {
        print("Upload failed: $responseBody");
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Failed: ${response.reasonPhrase}")),
        );
      }
    } catch (e) {
      print("Error: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error sending data")),
      );
    }
  }



  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Stack(
        children: [
          Center(
            child: SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _image != null
                      ? LayoutBuilder(
                    builder: (context, constraints) {
                      return FutureBuilder<Size>(
                        future: _getImageSize(_image!),
                        builder: (context, snapshot) {
                          if (!snapshot.hasData) return CircularProgressIndicator();

                          final imageSize = snapshot.data!;
                          final displayedWidth = constraints.maxWidth;
                          final displayedHeight = displayedWidth * imageSize.height / imageSize.width;

                          return Center(
                            child: SizedBox(
                              width: displayedWidth,
                              height: displayedHeight,
                              child: Stack(
                                fit: StackFit.expand,
                                children: [
                                  Image.file(_image!, fit: BoxFit.contain),
                                  if (_faces.isNotEmpty)
                                    CustomPaint(
                                      painter: FacePainter(_faces, imageSize, _isFrontCamera),
                                    ),
                                ],
                              ),
                            ),
                          );
                        },
                      );
                    },
                  )
                      : Icon(Icons.image, size: 150),

                  SizedBox(height: 20),

                  ElevatedButton(
                    onPressed: () async {
                      await captureAndDetect();
                    },
                    child: Text("Capture The Image"),
                  ),

                  if (_croppedFaces.isNotEmpty) ...[
                    SizedBox(height: 20),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16.0),
                      child: Wrap(
                        spacing: 10,
                        runSpacing: 10,
                        children: _croppedFaces.map((face) {
                          return Image.memory(
                            face.bytes,
                            width: 100,
                            height: 100,
                          );
                        }).toList(),
                      ),
                    ),

                    SizedBox(height: 20),

                    // Student ID input
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 40.0),
                      child: TextField(
                        controller: studentIdController,
                        decoration: InputDecoration(
                          border: OutlineInputBorder(),
                          labelText: 'Student ID',
                        ),
                      ),
                    ),

                    SizedBox(height: 10),

                    // Password input
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 40.0),
                      child: TextField(
                        controller: passwordController,
                        obscureText: true,
                        decoration: InputDecoration(
                          border: OutlineInputBorder(),
                          labelText: 'Password',
                        ),
                      ),
                    ),

                    SizedBox(height: 20),

                    ElevatedButton(
                      onPressed: () async {
                        if (_image == null) {
                          print("No image selected or captured");
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(content: Text("Please select or capture an image")),
                          );
                          return;
                        }

                        await sendDataToServer(
                          studentId: studentIdController.text,
                          password: passwordController.text,
                          imageFile: _image!,
                          faceEmbeddings: _faceEmbeddings,
                        );
                      },
                      child: Text('Send Embeddings'),
                    ),
                  ],
                ],
              ),
            ),
          ),

          if (_isProcessing)
            Container(
              color: Colors.black.withAlpha((0.5 * 255).round()),
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: const [
                    CircularProgressIndicator(),
                    SizedBox(height: 20),
                    Text(
                      "Processing...",
                      style: TextStyle(color: Colors.white, fontSize: 18),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
