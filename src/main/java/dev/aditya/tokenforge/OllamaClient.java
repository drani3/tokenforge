package dev.aditya.tokenforge;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Minimal client for a locally running Ollama server (https://ollama.com).
 * Ollama downloads and runs open-source models (Llama, Qwen, Gemma, Mistral, ...)
 * and exposes them over a simple HTTP API on localhost:11434.
 */
@Component
public class OllamaClient {

    public record Message(String role, String content) {}

    private final HttpClient http = HttpClient.newHttpClient();
    private final ObjectMapper json = new ObjectMapper();
    private final String baseUrl;
    private final String model;

    public OllamaClient(@Value("${ollama.base-url}") String baseUrl,
                        @Value("${ollama.model}") String model) {
        this.baseUrl = baseUrl;
        this.model = model;
    }

    public String model() {
        return model;
    }

    /**
     * Sends the whole conversation to the model and streams the reply token by token.
     * Returns the full reply so the caller can append it to the history.
     */
    public String chat(List<Message> messages, Consumer<String> onToken) throws Exception {
        String body = json.writeValueAsString(Map.of(
                "model", model,
                "messages", messages,
                "stream", true,
                "options", Map.of("temperature", 0.7)
        ));

        HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/api/chat"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<java.io.InputStream> response =
                http.send(request, HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            throw new IllegalStateException("Ollama returned " + response.statusCode() + ": "
                    + new String(response.body().readAllBytes(), StandardCharsets.UTF_8));
        }

        // Streaming responses are newline-delimited JSON, one chunk per line.
        StringBuilder reply = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(response.body(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                JsonNode chunk = json.readTree(line);
                String token = chunk.path("message").path("content").asText("");
                reply.append(token);
                onToken.accept(token);
                if (chunk.path("done").asBoolean()) break;
            }
        }
        return reply.toString();
    }
}
