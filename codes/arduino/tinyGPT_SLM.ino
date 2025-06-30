#include "gpt_model.h" // Include the generated header file

float start_time = -1;
float end_time = -1;
float width_time = -1;

// Define the buffer size for the model's response
// MAX_GENERATED_TEXT_LENGTH is defined in gpt_model.h, but you can use your own buffer here.
// Make sure it is large enough for the expected responses.
#define ANSWER_BUFFER_SIZE 256 
char answerBuffer[ANSWER_BUFFER_SIZE];

// Maximum number of new tokens to generate for the answer
int maxNewTokensForAnswer = 30; // Adjust as needed

void setup() {
  Serial.begin(115200); // Initialize serial communication
  while (!Serial) {
    ; // Wait for the serial connection to be established
  }
  Serial.println("Arduino Q&A model ready.");
  Serial.println("Type your question and press Enter:");
}

void loop() {
  if (Serial.available() > 0) {
    String question = Serial.readStringUntil('\n'); // Read the question from the Serial Monitor
    
    question.trim(); // Remove leading/trailing whitespace

    if (question.length() > 0) {
      Serial.print("Question: ");
      Serial.println(question);

      Serial.println("Generating answer...");

      // Call the Q&A generation function
      start_time = millis();
      // generate_qna_answer(char* output_buffer, int buffer_size, const String& question_prompt, int max_new_tokens)
      generate_qna_answer(answerBuffer, ANSWER_BUFFER_SIZE, question, maxNewTokensForAnswer);
      end_time = millis();
      
      Serial.print("Answer: ");
      Serial.println(answerBuffer);
      Serial.print("Processing time (ms): ");
      width_time = end_time - start_time;
      Serial.println(width_time);
      Serial.print("\nNext question: \n");
    }
  }
  delay(2000);
}