package com.oculus;

import android.os.Bundle;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraCharacteristics;
import android.content.Context;
import android.util.Log;

public class NativeActivity extends android.app.NativeActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // CMPUT428: Find the true Passthrough Camera IDs
        try {
            CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            for (String id : manager.getCameraIdList()) {
                CameraCharacteristics chars = manager.getCameraCharacteristics(id);

                // Meta's custom vendor keys
                CameraCharacteristics.Key<Integer> sourceKey = new CameraCharacteristics.Key<>("com.meta.extra_metadata.camera_source", int.class);
                CameraCharacteristics.Key<Integer> posKey = new CameraCharacteristics.Key<>("com.meta.extra_metadata.position", int.class);

                Integer source = chars.get(sourceKey);
                Integer pos = chars.get(posKey);

                // source == 0 means Passthrough RGB!
                if (source != null && source == 0) {
                    // pos == 0 is Left, pos == 1 is Right
                    String side = (pos != null && pos == 0) ? "Left" : "Right";
                    Log.v("CMPUT428", "TRUE " + side + " PASSTHROUGH ID: " + id);
                }
            }
        } catch (Exception e) {
            Log.e("CMPUT428", "Camera discovery failed", e);
        }
    }
}
