package com.finsecure.backend.security;

import io.jsonwebtoken.JwtException;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.lang.NonNull;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtService jwtService;
    private final CustomUserDetailsService userDetailsService;

    public JwtAuthenticationFilter(
            JwtService jwtService,
            CustomUserDetailsService userDetailsService) {

        this.jwtService = jwtService;
        this.userDetailsService = userDetailsService;
    }

    @Override
    protected void doFilterInternal(
            @NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response,
            @NonNull FilterChain filterChain)
            throws ServletException, IOException {

        String authHeader = request.getHeader("Authorization");

        System.out.println("=================================");
        System.out.println("REQUEST: " + request.getMethod()
                + " " + request.getRequestURI());

        System.out.println("AUTH HEADER PRESENT: "
                + (authHeader != null));

        // No Authorization header
        if (authHeader == null ||
                !authHeader.startsWith("Bearer ")) {

            System.out.println("NO VALID BEARER TOKEN FOUND");

            filterChain.doFilter(request, response);
            return;
        }

        // Extract JWT
        String jwt = authHeader.substring(7).trim();

        if (jwt.isEmpty()) {

            System.out.println("EMPTY JWT");

            filterChain.doFilter(request, response);
            return;
        }

        try {

            // Extract email from JWT
            String email = jwtService.extractUsername(jwt);

            System.out.println("JWT EMAIL: " + email);

            if (email != null &&
                    SecurityContextHolder.getContext()
                            .getAuthentication() == null) {

                // Find user in database
                UserDetails userDetails =
                        userDetailsService.loadUserByUsername(email);

                System.out.println(
                        "USER FOUND: " + userDetails.getUsername()
                );

                // Validate JWT
                boolean valid =
                        jwtService.isTokenValid(
                                jwt,
                                userDetails.getUsername()
                        );

                System.out.println(
                        "JWT VALID: " + valid
                );

                if (valid) {

                    UsernamePasswordAuthenticationToken authToken =
                            new UsernamePasswordAuthenticationToken(
                                    userDetails,
                                    null,
                                    userDetails.getAuthorities()
                            );

                    authToken.setDetails(
                            new WebAuthenticationDetailsSource()
                                    .buildDetails(request)
                    );

                    SecurityContextHolder
                            .getContext()
                            .setAuthentication(authToken);

                    System.out.println(
                            "AUTHENTICATION SUCCESSFUL: " + email
                    );

                } else {

                    System.out.println(
                            "JWT REJECTED: TOKEN INVALID OR EXPIRED"
                    );
                }
            }

        } catch (JwtException e) {

            System.out.println(
                    "JWT ERROR: " + e.getMessage()
            );

            SecurityContextHolder.clearContext();

        } catch (Exception e) {

            System.out.println(
                    "USER AUTHENTICATION ERROR: "
                            + e.getClass().getSimpleName()
                            + " - "
                            + e.getMessage()
            );

            SecurityContextHolder.clearContext();
        }

        filterChain.doFilter(request, response);
    }
}