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

        try {
            CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            String[] cameraIds = manager.getCameraIdList();
            Log.v("CMPUT428", "====== CAMERA DISCOVERY ======");
            Log.v("CMPUT428", "Found " + cameraIds.length + " cameras exposed to NDK.");

            for (String id : cameraIds) {
                Log.v("CMPUT428", "--- Inspecting Camera ID: " + id + " ---");
                CameraCharacteristics chars = manager.getCameraCharacteristics(id);

                // Check standard Android facing
                Integer facing = chars.get(CameraCharacteristics.LENS_FACING);
                String facingStr = (facing != null && facing == CameraCharacteristics.LENS_FACING_EXTERNAL) ? "EXTERNAL" : String.valueOf(facing);
                Log.v("CMPUT428", "Standard Lens Facing: " + facingStr);

                // Try to aggressively grab Meta's custom vendor keys
                try {
                    CameraCharacteristics.Key<Integer> sourceKey = new CameraCharacteristics.Key<>("com.meta.extra_metadata.camera_source", int.class);
                    Integer source = chars.get(sourceKey);
                    Log.v("CMPUT428", "Meta Source Key: " + (source != null ? source : "NULL"));
                } catch (IllegalArgumentException e) {
                    Log.v("CMPUT428", "Meta Source Key: NOT SUPPORTED ON THIS OS");
                }

                try {
                    CameraCharacteristics.Key<Integer> posKey = new CameraCharacteristics.Key<>("com.meta.extra_metadata.position", int.class);
                    Integer pos = chars.get(posKey);
                    Log.v("CMPUT428", "Meta Position Key: " + (pos != null ? pos : "NULL"));
                } catch (IllegalArgumentException e) {
                    Log.v("CMPUT428", "Meta Position Key: NOT SUPPORTED ON THIS OS");
                }
            }
            Log.v("CMPUT428", "==============================");
            
        } catch (Exception e) {
            Log.e("CMPUT428", "Camera discovery failed completely", e);
        }
    }
}