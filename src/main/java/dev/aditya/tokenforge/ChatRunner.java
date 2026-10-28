package dev.aditya.tokenforge;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Interactive terminal chat. Type a message, get a streamed reply.
 * Commands: /reset clears the conversation, /exit quits.
 */
@Component
@ConditionalOnProperty(name = "chat.enabled", havingValue = "true", matchIfMissing = true)
public class ChatRunner implements CommandLineRunner {

    private final OllamaClient ollama;
    private final String systemPrompt;

    public ChatRunner(OllamaClient ollama, @Value("${chat.system-prompt}") String systemPrompt) {
        this.ollama = ollama;
        this.systemPrompt = systemPrompt;
    }

    @Override
    public void run(String... args) throws Exception {
        List<OllamaClient.Message> history = newHistory();
        Scanner in = new Scanner(System.in);
        System.out.println("Chatting with " + ollama.model() + "  (/reset, /exit)");

        while (true) {
            System.out.print("\nyou> ");
            if (!in.hasNextLine()) break;
            String input = in.nextLine().trim();
            if (input.isEmpty()) continue;
            if (input.equals("/exit")) break;
            if (input.equals("/reset")) {
                history = newHistory();
                System.out.println("(conversation cleared)");
                continue;
            }

            history.add(new OllamaClient.Message("user", input));
            System.out.print("bot> ");
            try {
                String reply = ollama.chat(history, token -> {
                    System.out.print(token);
                    System.out.flush();
                });
                history.add(new OllamaClient.Message("assistant", reply));
                System.out.println();
            } catch (java.net.ConnectException e) {
                System.out.println("\nCan't reach Ollama. Is it running? Try `ollama serve`.");
                history.remove(history.size() - 1);
            }
        }
    }

    private List<OllamaClient.Message> newHistory() {
        List<OllamaClient.Message> history = new ArrayList<>();
        history.add(new OllamaClient.Message("system", systemPrompt));
        return history;
    }
}
