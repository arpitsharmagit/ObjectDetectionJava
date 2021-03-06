package com.gl;
public class RunStats implements AutoCloseable {
    private long nativeHandle = allocate();
    private static byte[] fullTraceRunOptions = new byte[]{8, 3};

    public static byte[] runOptions() {
        return fullTraceRunOptions;
    }

    public RunStats() {
    }

    public void close() {
        if(this.nativeHandle != 0L) {
            delete(this.nativeHandle);
        }

        this.nativeHandle = 0L;
    }

    public synchronized void add(byte[] var1) {
        add(this.nativeHandle, var1);
    }

    public synchronized String summary() {
        return summary(this.nativeHandle);
    }

    private static native long allocate();

    private static native void delete(long var0);

    private static native void add(long var0, byte[] var2);

    private static native String summary(long var0);
}