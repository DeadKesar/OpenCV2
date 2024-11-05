using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Windows;
using System.Windows.Media.Imaging;

namespace SURF
{
    public static class BitmapSourceConvert
    {
        [System.Runtime.InteropServices.DllImport("gdi32.dll")]
        public static extern bool DeleteObject(IntPtr hObject);

        public static BitmapSource ToBitmapSource(Mat mat)
        {
            using (var source = mat.ToBitmap())
            {
                IntPtr ptr = source.GetHbitmap();
                BitmapSource bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                    ptr, IntPtr.Zero, Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
                DeleteObject(ptr);
                return bs;
            }
        }

        public static BitmapSource ToBitmapSource(Image<Bgr, Byte> image)
        {
            using (var source = image.ToBitmap())
            {
                IntPtr ptr = source.GetHbitmap();
                BitmapSource bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                    ptr, IntPtr.Zero, Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
                DeleteObject(ptr);
                return bs;
            }
        }
    }
}