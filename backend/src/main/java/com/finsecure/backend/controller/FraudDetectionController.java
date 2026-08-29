package com.finsecure.backend.controller;

import com.finsecure.backend.dto.FraudDetectionRequest;
import com.finsecure.backend.dto.FraudDetectionResponse;
import com.finsecure.backend.service.FraudDetectionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/fraud")
public class FraudDetectionController {

    private final FraudDetectionService fraudDetectionService;

    public FraudDetectionController(
            FraudDetectionService fraudDetectionService) {

        this.fraudDetectionService = fraudDetectionService;
    }

    @PostMapping("/predict")
    public ResponseEntity<FraudDetectionResponse> predictFraud(
            @RequestBody FraudDetectionRequest request) {

        FraudDetectionResponse response =
                fraudDetectionService.predictFraud(request);

        return ResponseEntity.ok(response);
    }
}