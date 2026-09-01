// Prediction-serving microservice. Loads predictions produced offline by the
// Python/PyTorch/XGBoost pipelines (api/go/export_predictions.py) and serves
// them over HTTP with goroutine-per-request concurrency. This service does
// NOT run model inference itself — that split (train/score offline in
// Python, serve online in Go) is deliberate: Go buys request throughput
// here, not numerical compute, which is what the rest of this repo already
// does well in Python/C++.
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

type CreditPrediction struct {
	ApplicantID            int     `json:"applicant_id"`
	PDScore                float64 `json:"pd_score"`
	DTI                    float64 `json:"dti"`
	NumPriorDelinquencies  int     `json:"num_prior_delinquencies"`
	ActualDefault          int     `json:"actual_default"`
}

type EquitySnapshot struct {
	AsOf                  string  `json:"as_of"`
	Close                 float64 `json:"close"`
	LogReturn             float64 `json:"log_return"`
	SMA20                 float64 `json:"sma_20"`
	RealizedVol20d        float64 `json:"realized_vol_20d"`
	LSTMTestAccuracy      float64 `json:"lstm_test_accuracy"`
	MajorityClassBaseline float64 `json:"majority_class_baseline"`
}

type Predictions struct {
	CreditPredictions []CreditPrediction `json:"credit_predictions"`
	EquitySnapshot    EquitySnapshot     `json:"equity_snapshot"`
}

type Store struct {
	mu   sync.RWMutex
	data Predictions
}

func (s *Store) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var p Predictions
	if err := json.NewDecoder(f).Decode(&p); err != nil {
		return err
	}

	s.mu.Lock()
	s.data = p
	s.mu.Unlock()
	return nil
}

func (s *Store) Snapshot() Predictions {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.data
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func main() {
	dataPath := "api/go/data/predictions.json"
	if len(os.Args) > 1 {
		dataPath = os.Args[1]
	}

	store := &Store{}
	if err := store.Load(dataPath); err != nil {
		log.Fatalf("failed to load predictions from %s: %v", dataPath, err)
	}
	log.Printf("loaded predictions from %s", dataPath)

	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	mux.HandleFunc("GET /v1/credit/predictions", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, store.Snapshot().CreditPredictions)
	})

	mux.HandleFunc("GET /v1/equity/snapshot", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, store.Snapshot().EquitySnapshot)
	})

	addr := ":8080"
	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
	}

	log.Printf("listening on %s", addr)
	log.Fatal(srv.ListenAndServe())
}
