package com.finsecure.backend.repository;

import com.finsecure.backend.entity.FraudTransaction;
import com.finsecure.backend.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface FraudTransactionRepository
        extends JpaRepository<FraudTransaction, Long> {

    List<FraudTransaction> findByUserOrderByCreatedAtDesc(User user);
}
