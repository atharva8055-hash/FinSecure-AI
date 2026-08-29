package com.finsecure.backend.service;

import com.finsecure.backend.dto.FraudDetectionRequest;
import com.finsecure.backend.dto.FraudDetectionResponse;
import com.finsecure.backend.entity.FraudTransaction;
import com.finsecure.backend.entity.User;
import com.finsecure.backend.repository.FraudTransactionRepository;
import com.finsecure.backend.repository.UserRepository;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;

@Service
public class FraudDetectionService {

    private final RestTemplate restTemplate;
    private final FraudTransactionRepository fraudTransactionRepository;
    private final UserRepository userRepository;

    private static final String ML_API_URL =
            "http://127.0.0.1:5000/predict";

    public FraudDetectionService(
            FraudTransactionRepository fraudTransactionRepository,
            UserRepository userRepository) {

        this.restTemplate = new RestTemplate();
        this.fraudTransactionRepository = fraudTransactionRepository;
        this.userRepository = userRepository;
    }

    public FraudDetectionResponse predictFraud(
            FraudDetectionRequest request) {

        // Send transaction to Python ML API
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<FraudDetectionRequest> entity =
                new HttpEntity<>(request, headers);

        FraudDetectionResponse response =
                restTemplate.postForObject(
                        ML_API_URL,
                        entity,
                        FraudDetectionResponse.class
                );

        // Get currently authenticated user
        Authentication authentication =
                SecurityContextHolder.getContext().getAuthentication();

        if (authentication == null ||
                authentication.getName() == null) {

            throw new RuntimeException(
                    "Authenticated user not found");
        }

        String email = authentication.getName();

        User user = userRepository.findByEmail(email)
                .orElseThrow(() ->
                        new RuntimeException(
                                "User not found: " + email));

        // Save fraud prediction
        FraudTransaction transaction =
                new FraudTransaction();

        transaction.setUser(user);
        transaction.setStep(request.getStep());
        transaction.setAmount(request.getAmount());
        transaction.setOldbalanceOrg(request.getOldbalanceOrg());
        transaction.setOldbalanceDest(request.getOldbalanceDest());
        transaction.setType(request.getType());

        if (response != null &&
                response.getPrediction() != null) {

            FraudDetectionResponse.Prediction prediction =
                    response.getPrediction();

            transaction.setFraudProbability(
                    prediction.getFraud_probability());

            transaction.setRiskScore(
                    prediction.getRisk_score());

            transaction.setRiskLevel(
                    prediction.getRisk_level());

            transaction.setFraud(
                    prediction.isIs_fraud());

            transaction.setDecision(
                    prediction.getDecision());
        }

        transaction.setCreatedAt(LocalDateTime.now());

        fraudTransactionRepository.save(transaction);

        return response;
    }
}