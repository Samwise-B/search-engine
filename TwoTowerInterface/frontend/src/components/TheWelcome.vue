<template>
  <div class="container">
    <!-- Input field for string -->
    <input v-model="userInput" type="text" placeholder="Enter your message" class="input-box" />

    <!-- Send button -->
    <button @click="sendMessage" class="send-button">Send</button>

    <!-- Display the submitted message -->
    <p v-if="message" class="message">{{ message }}</p>
    <!-- Display the list of passages -->
     <h1 v-if="initQuery">Query: {{ initQuery }}</h1>
    <p v-for="(passage, index) in passages" :key="index">{{ passage }}</p>
  </div>
</template>

<script>
import { ref } from 'vue'
import axios from 'axios';

export default {
  setup() {
    // Reactive references for input, message, and passages
    const userInput = ref('')  // To store the input from the text field
    const message = ref('')    // To store the message to be displayed
    const passages = ref([])   // To store the list of passages returned from the backend
    const initQuery = ref('')  // To store the initial query to be sent to the backend
    // Function to send the message to the backend and update passages
    const sendMessage = async () => {
      try {
        initQuery.value = userInput.value;
        // Make a POST request to FastAPI backend
        const response = await axios.post('http://127.0.0.1:8000/api/user_input', {
        query: userInput.value  // Send the input as the 'query' field in the request body
        });
        console.log(response.data);
        passages.value = response.data;

        userInput.value = '';

      } catch (error) {
        console.error(error);  // Handle error response
        alert("Error sending message.");
      }
    };

    return {
      userInput,
      initQuery,
      message,
      passages,
      sendMessage
    };
  }
}
</script>

<style scoped>
/* Centering the content */
.container {
  padding: 20px;
  align-items: center;
  justify-content: center;
  margin-top: 40px;
}

/* Input field styling */
.input-box {
  margin-right: 5px;
  width: 100%;
  max-width: 400px;
  padding: 12px;
  font-size: 18px;
  border-radius: 8px;
  border: 1px solid #ccc;
  outline: none;
  transition: border-color 0.3s ease;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.input-box:focus {
  border-color: #3f51b5;
  box-shadow: 0 2px 6px rgba(63, 81, 181, 0.2);
}

/* Button styling */
.send-button {
  margin-top: 12px;
  padding: 12px 20px;
  font-size: 18px;
  background-color: #3f51b5;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s ease, transform 0.2s;
}

.send-button:hover {
  background-color: #303f9f;
  transform: translateY(-2px);
}

.send-button:active {
  background-color: #303f9f;
  transform: translateY(0);
}

/* Message display styling */
.message {
  margin-top: 20px;
  font-size: 20px;
  color: #333;
  font-weight: 500;
}
</style>
