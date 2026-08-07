package com.finsecure.backend.controller;

import com.finsecure.backend.dto.LoginRequest;
import com.finsecure.backend.dto.LoginResponse;
import com.finsecure.backend.dto.RegisterRequest;
import com.finsecure.backend.service.AuthService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService authService;

    // Constructor Injection
    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    // ==========================
    // REGISTER API
    // ==========================

    @PostMapping("/register")
    public String register(@RequestBody RegisterRequest request) {
        return authService.register(request);
    }

    // ==========================
    // LOGIN API
    // ==========================

    @PostMapping("/login")
    public LoginResponse login(@RequestBody LoginRequest request) {
        return authService.login(request);
    }
}