using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.Win32;
using System;
using System.Windows;
using System.Windows.Threading;

namespace SURF
{
    public partial class MainWindow : Window
    {
        private Image<Bgr, Byte> imgSceneColor;
        private Image<Bgr, Byte> imgToFindColor;
        private VideoCapture _capture;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void ButtonStartCamera_Click(object sender, RoutedEventArgs e)
        {
            if (_capture == null)
            {
                _capture = new VideoCapture(0);
                _capture.ImageGrabbed += ProcessFrame;
            }
            _capture.Start();
        }

        private void ButtonLoadTemplate_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                imgToFindColor = new Image<Bgr, Byte>(openFileDialog.FileName);
            }
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            Mat frame = new Mat();
            _capture.Retrieve(frame);

            if (frame.IsEmpty)
                return;

            imgSceneColor = frame.ToImage<Bgr, Byte>();

            ProcessImage();
        }

        private void ProcessImage()
        {
            if (imgSceneColor == null || imgToFindColor == null)
            {
                return;
            }

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

            VectorOfKeyPoint vkpSceneKeyPoints = new VectorOfKeyPoint();
            VectorOfKeyPoint vkpToFindKeyPoints = new VectorOfKeyPoint();
            Mat mtxSceneDescriptors = new Mat();
            Mat mtxToFindDescriptors = new Mat();

            orbDetector.DetectAndCompute(imgSceneGray, null, vkpSceneKeyPoints, mtxSceneDescriptors, false);
            orbDetector.DetectAndCompute(imgToFindGray, null, vkpToFindKeyPoints, mtxToFindDescriptors, false);

            BFMatcher matcher = new BFMatcher(DistanceType.Hamming);
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            matcher.KnnMatch(mtxToFindDescriptors, mtxSceneDescriptors, matches, 2);

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

            Mat resultImg = new Mat();
            if (goodMatches.Size >= 4)
            {
                Features2DToolbox.DrawMatches(
                    imgToFindColor,
                    vkpToFindKeyPoints,
                    imgSceneColor,
                    vkpSceneKeyPoints,
                    goodMatches,
                    resultImg,
                    new MCvScalar(0, 255, 0),
                    new MCvScalar(255, 0, 0),
                    null,
                    Features2DToolbox.KeypointDrawType.Default);

                Dispatcher.Invoke(() =>
                {
                    ImageResult.Source = BitmapSourceConvert.ToBitmapSource(resultImg);
                });
            }
            else
            {
                Dispatcher.Invoke(() =>
                {
                    ImageResult.Source = BitmapSourceConvert.ToBitmapSource(imgSceneColor);
                });
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);

            if (_capture != null)
            {
                _capture.ImageGrabbed -= ProcessFrame;
                _capture.Dispose();
            }
        }
    }
}
