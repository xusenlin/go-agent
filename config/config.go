package config

import (
	"bufio"
	"os"
	"strings"
)

// LoadEnv loads environment variables from .env file
func LoadEnv() {
	loadEnvFile(".env")
}

// loadEnvFile reads a .env file and sets environment variables
func loadEnvFile(filename string) {
	file, err := os.Open(filename)
	if err != nil {
		return // File doesn't exist, that's fine
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse key=value
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		// Only set if not already set in environment
		if os.Getenv(key) == "" {
			os.Setenv(key, value)
		}
	}
}

// GetEnv returns environment variable with fallback
func GetEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

// GetOpenAIConfig returns OpenAI configuration from environment
func GetOpenAIConfig() (apiKey, model, baseURL string) {
	apiKey = GetEnv("OPENAI_API_KEY", "")
	model = GetEnv("OPENAI_MODEL", "gpt-4o")
	baseURL = GetEnv("OPENAI_BASE_URL", "https://api.openai.com/v1")
	return
}
