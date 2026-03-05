// ALICE-History — Unity C# バインディング
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Moroya Sakamoto
//
// 10 FFI 関数 (Fragment 3 + InversionConfig 2 + restore 1 + RestorationResult 4)

using System;
using System.Runtime.InteropServices;

namespace Alice.History
{
    /// <summary>フラグメント種別。</summary>
    public enum FragmentKind : byte
    {
        Text        = 0,
        Image       = 1,
        Artifact    = 2,
        Inscription = 3,
        Audio       = 4,
    }

    /// <summary>歴史フラグメント。</summary>
    public sealed class AliceFragment : IDisposable
    {
        private IntPtr _ptr;

        public AliceFragment(ulong id, FragmentKind kind, double[] data, double[] mask, ulong timestampNs)
        {
            unsafe
            {
                fixed (double* dp = data)
                fixed (double* mp = mask)
                {
                    _ptr = NativeMethods.alice_fragment_new(
                        id, (byte)kind, (IntPtr)dp, (IntPtr)mp, (uint)data.Length, timestampNs);
                }
            }
        }

        internal IntPtr Ptr => _ptr;

        /// <summary>既知データの割合。</summary>
        public double KnownFraction()
        {
            if (_ptr == IntPtr.Zero) return 0.0;
            return NativeMethods.alice_fragment_known_fraction(_ptr);
        }

        public void Dispose()
        {
            if (_ptr != IntPtr.Zero)
            {
                NativeMethods.alice_fragment_destroy(_ptr);
                _ptr = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~AliceFragment() => Dispose();
    }

    /// <summary>復元設定。</summary>
    public sealed class AliceInversionConfig : IDisposable
    {
        private IntPtr _ptr;

        public AliceInversionConfig()
        {
            _ptr = NativeMethods.alice_config_default();
        }

        internal IntPtr Ptr => _ptr;

        public void Dispose()
        {
            if (_ptr != IntPtr.Zero)
            {
                NativeMethods.alice_config_destroy(_ptr);
                _ptr = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~AliceInversionConfig() => Dispose();
    }

    /// <summary>復元結果。</summary>
    public sealed class AliceRestorationResult : IDisposable
    {
        private IntPtr _ptr;

        internal AliceRestorationResult(IntPtr ptr) { _ptr = ptr; }

        /// <summary>復元値配列を取得。</summary>
        public double[] GetValues()
        {
            if (_ptr == IntPtr.Zero) return Array.Empty<double>();
            uint len = NativeMethods.alice_result_values_len(_ptr);
            if (len == 0) return Array.Empty<double>();
            IntPtr src = NativeMethods.alice_result_values_ptr(_ptr);
            double[] arr = new double[len];
            Marshal.Copy(src, arr, 0, (int)len);
            return arr;
        }

        /// <summary>復元前エントロピー。</summary>
        public double EntropyBefore
        {
            get
            {
                if (_ptr == IntPtr.Zero) return 0.0;
                return NativeMethods.alice_result_entropy_before(_ptr);
            }
        }

        /// <summary>反復回数。</summary>
        public uint Iterations
        {
            get
            {
                if (_ptr == IntPtr.Zero) return 0;
                return NativeMethods.alice_result_iterations(_ptr);
            }
        }

        public void Dispose()
        {
            if (_ptr != IntPtr.Zero)
            {
                NativeMethods.alice_result_destroy(_ptr);
                _ptr = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~AliceRestorationResult() => Dispose();
    }

    /// <summary>復元 API。</summary>
    public static class AliceRestore
    {
        /// <summary>フラグメントを復元する。</summary>
        public static AliceRestorationResult Restore(AliceFragment fragment, AliceInversionConfig config)
        {
            IntPtr ptr = NativeMethods.alice_restore(fragment.Ptr, config.Ptr);
            if (ptr == IntPtr.Zero) return null;
            return new AliceRestorationResult(ptr);
        }
    }

    // ── P/Invoke ────────────────────────────────────────────────────
    internal static class NativeMethods
    {
        private const string Lib = "alice_history";

        // Fragment (3)
        [DllImport(Lib)] public static extern IntPtr alice_fragment_new(
            ulong id, byte kind, IntPtr data_ptr, IntPtr mask_ptr, uint len, ulong timestamp_ns);
        [DllImport(Lib)] public static extern double alice_fragment_known_fraction(IntPtr fragment);
        [DllImport(Lib)] public static extern void alice_fragment_destroy(IntPtr fragment);

        // InversionConfig (2)
        [DllImport(Lib)] public static extern IntPtr alice_config_default();
        [DllImport(Lib)] public static extern void alice_config_destroy(IntPtr config);

        // restore (1)
        [DllImport(Lib)] public static extern IntPtr alice_restore(IntPtr fragment, IntPtr config);

        // RestorationResult (4)
        [DllImport(Lib)] public static extern IntPtr alice_result_values_ptr(IntPtr result);
        [DllImport(Lib)] public static extern uint alice_result_values_len(IntPtr result);
        [DllImport(Lib)] public static extern double alice_result_entropy_before(IntPtr result);
        [DllImport(Lib)] public static extern uint alice_result_iterations(IntPtr result);
        [DllImport(Lib)] public static extern void alice_result_destroy(IntPtr result);
    }
}
