using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.Win32;
using System;
using System.Windows;
using System.Windows.Media.Imaging;

namespace SURF
{
    public partial class MainWindow : Window
    {
        private VideoCapture _capture;
        private Mat _currentFrame;
        private Image<Bgr, Byte> imgToFindColor;

        public MainWindow()
        {
            InitializeComponent();
            _currentFrame = new Mat();
        }

        private void ButtonLoadScene_Click(object sender, RoutedEventArgs e)
        {
            if (_capture == null)
            {
                _capture = new VideoCapture();
                _capture.ImageGrabbed += ProcessFrame;
                _capture.Start();
            }
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            _capture.Retrieve(_currentFrame);
            Dispatcher.Invoke(() =>
            {
                ImageScene.Source = BitmapSourceConvert.ToBitmapSource(_currentFrame);
            });
        }

        private void ButtonLoadTemplate_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                imgToFindColor = new Image<Bgr, Byte>(openFileDialog.FileName);
                ImageResult.Source = BitmapSourceConvert.ToBitmapSource(imgToFindColor);
            }
        }

        private void ButtonProcess_Click(object sender, RoutedEventArgs e)
        {
            if (_currentFrame == null || imgToFindColor == null)
            {
                MessageBox.Show("Please load the template image and start the camera.");
                return;
            }

            Image<Bgr, Byte> imgSceneColor = _currentFrame.ToImage<Bgr, Byte>();

            // Создание ORB-детектора с указанными параметрами
            var orbDetector = new ORB(
                numberOfFeatures: 500,
                scaleFactor: 1.2f,
                nLevels: 8,
                edgeThreshold: 31,
                firstLevel: 0,
                WTK_A: 2,
                patchSize: 31,
                fastThreshold: 20
            );

            Image<Gray, Byte> imgSceneGray = imgSceneColor.Convert<Gray, Byte>();
            Image<Gray, Byte> imgToFindGray = imgToFindColor.Convert<Gray, Byte>();

            // Выделение ключевых точек и создание дескрипторов
            VectorOfKeyPoint vkpSceneKeyPoints = new VectorOfKeyPoint();
            VectorOfKeyPoint vkpToFindKeyPoints = new VectorOfKeyPoint();
            Mat mtxSceneDescriptors = new Mat();
            Mat mtxToFindDescriptors = new Mat();

            orbDetector.DetectAndCompute(imgSceneGray, null, vkpSceneKeyPoints, mtxSceneDescriptors, false);
            orbDetector.DetectAndCompute(imgToFindGray, null, vkpToFindKeyPoints, mtxToFindDescriptors, false);

            // Сопоставление с помощью BFMatcher
            BFMatcher matcher = new BFMatcher(DistanceType.Hamming);
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            matcher.KnnMatch(mtxSceneDescriptors, mtxToFindDescriptors, matches, 2);

            // Фильтрация совпадений по уникальности
            float uniquenessThreshold = 0.8f;
            VectorOfDMatch goodMatches = new VectorOfDMatch();
            for (int i = 0; i < matches.Size; i++)
            {
                var match = matches[i];
                if (match.Size >= 2 && match[0].Distance < uniquenessThreshold * match[1].Distance)
                {
                    goodMatches.Push(new MDMatch[] { match[0] });
                }
            }

            // Отрисовка результата
            Mat resultImg = new Mat();
            if (goodMatches.Size >= 4)
            {
                // Используем пустую маску, чтобы отобразить только хорошие совпадения
                Features2DToolbox.DrawMatches(
                    imgToFindColor,
                    vkpToFindKeyPoints,
                    imgSceneColor,
                    vkpSceneKeyPoints,
                    goodMatches,
                    resultImg,
                    new MCvScalar(0, 255, 0), // Цвет совпадающих линий
                    new MCvScalar(255, 0, 0), // Цвет одиночных точек
                    null, // Используем null вместо matchesMask
                    Features2DToolbox.KeypointDrawType.Default);

                ImageResult.Source = BitmapSourceConvert.ToBitmapSource(resultImg);
            }
            else
            {
                MessageBox.Show("Not enough matches found!");
            }
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            if (_capture != null)
            {
                _capture.ImageGrabbed -= ProcessFrame;
                _capture.Dispose();
            }
            base.OnClosing(e);
        }
    }
}