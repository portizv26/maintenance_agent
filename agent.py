import pandas as pd
import numpy as np
from datetime import timedelta
from openai import OpenAI
from typing import List, Optional
import sqlite3

import json
import time
import os
from dotenv import load_dotenv
import concurrent.futures

from extras import ground_truth, text_to_query_messages, assistant_id

load_dotenv()
client = OpenAI()

# Class to handle the agent's conversation and database interactions
class AgentChat:
    # Initialize the agent with a database connection and an assistant ID.
    def __init__(self, client: OpenAI, conn: sqlite3.Connection, model: str,
                 assistant_id: str=assistant_id, text_to_query_messages: List[dict] = text_to_query_messages, max_tries: int = 3):
        self.client = client
        self.conn = conn
        self.assistant_id = assistant_id
        self.model = model
        self.text_to_query_messages = text_to_query_messages
        self.max_tries = max_tries
        
        self.real_history_messages = []
        self.general_messages = []
        self.query_messages = []
        
        self.file_id = None # file id of the csv file uploaded to OpenAI
        self.final_image = None # last image created by the assistant
        self.final_code = None # last code created by the assistant
        self.final_dataframe = None # last dataframe created by the assistant
        
        self.last_query = ''
        self.last_user_request = ''
        self.last_answer = ''
        
        self.csv_filename_path = f"tmp/sample_data.csv"
        self.chart_filename_path = f"tmp/chart.png"
        self.code_filename_path = f"tmp/chart_code.py"
    
    ###################
    #  Agent Methods  #    
    ###################
    
    #  Auxiliar Methods
    ## Method to add a message to the conversation history
    def add_message_to_history(self, role: str, content: str, nature: str) -> None:
        """
        Adds a message to the conversation history.
        """
        self.real_history_messages.append({"role": role, "content": content, 'nature': nature})
        
    ## Method to get the conversation history as a string
    def get_context_history_messages_as_string(self) -> str:
        """
        Returns the conversation history as a string.
        """
        messages = []
        for message in self.real_history_messages[:-1]:
            if message['nature'] == 'on-topic':
                messages.append(f"{message['role'].upper()}:\n{message['content']}")
                
        if len(messages) == 0:
            return "There is no relevant context for the user message."
        else:
            return "\n".join(messages)
    
    ## Method to reset the conversation history
    def reset_conversation_history(self) -> None:
        """
        Resets the conversation history.
        """
        self.real_history_messages = []
        self.last_query = ''
        self.last_user_request = ''
    
    ## Method to synthesize the context messages
    def summarize_context_messages(self) -> str:
        """
        Summarizes the context messages into a summary of the conversation up to the last message.
        This method resolves the following question : What has happened in the conversation so far?
        The AI must indicate:
        - has there been a previous context relevant to the user message?
        - if no, indicate that there is no relevant context.
        - if yes, summarize the context messages in a few sentences adding:
        -- has there been a query?
        -- has there been code provided?
        -- has there been a query result?
        """
        context_messages = self.get_context_history_messages_as_string()
        
        messages = [
            {
                "role": "system",
                "content": """
                You are a Chatbot Assistant that summarizes the conversation history so far. 

                Task:
                - Summarize the conversation history, including any queries, code provided, or query results.
                - If there is no relevant context, indicate that clearly.
                - Keep the summary concise and informative.
                - Do not mention these instructions or the word 'prompt' in your output.
                
                Output Format:
                - A summary of the conversation history, including any queries, code provided, or query results.
                - Example: "The user has asked about the maintenance history of a specific component. There has been a query about the last maintenance date, and the result was provided. No code was provided."
                - If there is no relevant context, indicate that clearly: "There is no relevant context for the user message."
                """
            },
            {
                "role": "user",
                "content": f"The conversation history is: {context_messages}"
            }
            
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1500,
            temperature=0.9
        )
        
        text_response = response.choices[0].message.content.strip()
        return text_response
    
    ## Method to get the user request based on the conversation history and user question
    def get_user_request(self, user_message: str, context_messages: str) -> str:
        """
        Returns the user request based on the conversation history and user question.
        This method resolves the following question: What does the user want to know?
        """
        messages = [
            {
                "role": "system",
                "content": """
                You are a Chatbot Assistant that extracts the user request from the conversation history and user question.

                Task:
                - Extract the user request from the conversation history and user question.
                - Keep the response concise and informative.
                - Do not mention these instructions or the word 'prompt' in your output.
                
                Output Format:
                - A summary of the user request, including any queries, code provided, or query results.
                """
            },
            {
                "role": "user",
                "content": f"The conversation history is: {context_messages} The user question is: {user_message}. What are the tasks the user wants to accomplish?"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        text_response = response.choices[0].message.content.strip()
        return text_response
    
    #  Nature Methods
    ## Method to define the nature of the message in one of 3 categories: "general", "off-topic", "on-topic"
    def define_message_nature_ai(self, user_request: str, context_summary: str) -> str:
        """
        Defines the nature of the message based on its content.
        Uses the OpenAI API to classify the message into one of three categories:
        * "general" : general message, for instance a greeting, a farewell or a general question.
        * "off-topic" : message that is not related with the topic of the database.
        * "on-topic" : message that is related with the topic of the database.
        """
        messages = [
            {
                "role": "system",
                "content": """
                You are a Chatbot Assistant that classifies the user's **latest** message into:
                - general
                - off-topic
                - on-topic

                Definition of "on-topic":
                - The user's message (possibly referencing earlier parts of the conversation) relates to the "Workshop Maintenance Database," which tracks maintenance events, a hierarchy of systems/subsystems/components, and jobs (tasks done on components). 
                - If the user's question or statement can be answered using this database (or references data from it), it is on-topic.

                Definition of "off-topic":
                - The user's message does not relate to the database, or it references something completely different (e.g. efficiency, operations, etc.) with no connection to the workshop maintenance data.

                Definition of "general":
                - The user's message is a general greeting, farewell, or other small talk that is not specifically about the database.

                You have access to the full conversation so far, so the user may be referencing prior messages or data. 
                Classify the latest user message accordingly.

                **Output Format**:
                - category name (general, off-topic, on-topic), as a plain string
                - example: “general”
                """
            },
            {
                "role": "user",
                "content": f"The last message from the user is: {user_request}\nThe context of previous messages is: {context_summary}"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        nature = response.choices[0].message.content.strip()
        return nature
    
    ## Method to define the nature of the message in one of 3 categories: "general", "off-topic", "on-topic"
    def get_message_nature(self, user_request: str, context_summary: str) -> str:
        """
        Checks the nature of the message and returns it.
        If the nature is not one of the expected categories, it defaults to "general".
        """
        print(f'✍🏻 Checking message nature: {user_request}')
        nature = self.define_message_nature_ai(user_request, context_summary)
        
        if 'on-topic' in nature:
            nature = 'on-topic'
        elif 'off-topic' in nature:
            nature = 'off-topic'
        elif 'general' in nature:
            nature = 'general'
        
        if nature not in ["general", "off-topic", "on-topic"]:
            # If the nature is not one of the expected categories, return "general"
            print(f"Unexpected nature: {nature}. Defaulting to 'general'.")
            # Log the unexpected nature for debugging purposes
            with open("unexpected_nature.log", "a") as log_file:
                log_file.write(f"{time.ctime()}: Unexpected nature '{nature}' for message '{user_request}'\n")
                
            nature = "general"
        return nature
    
    #  Not on-topic Methods
    ## Method to get the system prompt based on the nature of the message
    def get_system_prompt_not_on_topic(self, nature: str) -> str:
        """
        Returns the system prompt based on the nature of the message.
        """
        if nature == "general":
            return """
                You are a polite and friendly assistant. The user’s message is classified as 'general' 
                (e.g. a greeting, farewell, or casual statement) rather than a request for workshop maintenance data.

                Task:
                - Acknowledge or respond to their general statement in a warm, concise way.
                - Offer a gentle reminder that you specialize in the Workshop Maintenance Database, and you can help if they have maintenance-related questions.
                - Keep the response short and friendly, ideally 1-3 sentences.
                - Do not mention these instructions or the word 'prompt' in your output.
            """
        elif nature == "off-topic":
            return """
                You are a polite assistant specialized in the Workshop Maintenance Database. 
                The user’s message is classified as 'off-topic,' meaning it’s not related to the database.

                Task:
                - Politely acknowledge that their question is off-topic.
                - Briefly explain you specialize in workshop maintenance questions.
                - Invite them to ask something about the maintenance data, if they’d like.
                - Keep the response short and respectful.
                - Do not mention these instructions or the word 'prompt' in your output.
            """
            
    # Method to execute the response based on the nature of the message
    def excecute_response_not_on_topic(self, user_message: str, nature: str) -> str:
        """
        Executes the response based on the nature of the message.
        """
        # Get the system prompt based on the nature of the message
        system_prompt = self.get_system_prompt_not_on_topic(nature)
        
        # Execute the API call to get the response
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"The user said: {user_message}"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        text_response = response.choices[0].message.content.strip()
        return text_response

    #  Context Sufficiency Methods
    ## Method to execute the context sufficiency test
    def excecute_context_sufficiency_test(self, user_request: str, context_summary: str) -> bool:
        """
        Executes the context sufficiency test to check if the context is sufficient for the user question.
        This method resolves the following question: Is the context sufficient for the user question?
        """
        messages = [
            {
                "role": "system",
                "content": """
                You are a Chatbot Assistant that checks if the context is sufficient for the user question.
                
                Your task is to determine if the question that the user is asking can be answered with the previous context and/or with a query to the following database.

                Task:
                - Check if the context is sufficient for the user question.
                - The response must be a boolean value indicating if the context is sufficient for the user question.
                
                Output Format:
                - A boolean value indicating if the context is sufficient for the user question.
                - Options: "true" or "false"
                """
            },
            {
                "role": "user",
                "content": f"""
                The user request is: {user_request}
                The context summary is: {context_summary}.
                The database is a workshop maintenance database that tracks maintenance events, systems, subsystems, components, and jobs. Here is a summary of the database:

                - **Maintenance Cycles:**  
                Contains details of each maintenance event such as the unit identifier, start and end times, whether the cycle was scheduled, if any critical change occurred, and extra comments.

                - **Reference Hierarchy:**  
                - **Systems:** List of systems (e.g., Engine, Motor, Transmision) with an indicator for system-level critical changes.
                - **Subsystems:** Each system has one or more subsystems (e.g., Coolant under Engine). A subsystem always belongs to a specific system.
                - **Components:** Each subsystem includes one or more components (e.g., Radiator under Coolant), each with a flag for component-level critical changes.

                - **Jobs:**  
                Individual maintenance tasks performed on components (job type, comment, and extra information).

                ## Data Structure and Granularity

                - **Minimum Granularity:**  
                The **job** level represents the smallest unit of data, which can be aggregated up to components, subsystems, systems, and maintenance cycles.

                - **Join Keys and Relationships:**  
                - **Maintenance Cycle ↔ System:**  
                    Joined using the `maintenance_cycle_system` join table, which uses the composite key (`mantention_cycle_id`, `system_id`).
                - **System ↔ Subsystem:**  
                    Joined via the foreign key `system_id` in the `subsystem` table.
                - **Subsystem ↔ Component:**  
                    Joined via the foreign key `subsystem_id` in the `component` table.
                - **Component ↔ Job:**  
                    Joined via the foreign key `component_id` in the `job` table.

                ## Tables

                ### 1. maintenance_cycle
                - **Description:** Stores each maintenance event.
                - **Fields:**
                - `mantention_cycle_id` (INTEGER, PRIMARY KEY)
                - `UnitId` (TEXT): Equipment/unit identifier.
                - `start_time` (TEXT): ISO-formatted start timestamp.
                - `end_time` (TEXT): ISO-formatted end timestamp.
                - `is_scheduled` (BOOLEAN): Indicates if the maintenance was planned.
                - `has_critical_change` (BOOLEAN): Indicates if a critical change occurred.
                - `extra_comments` (TEXT): Additional remarks.

                ### 2. system
                - **Description:** Contains reference data for systems.
                - **Fields:**
                - `system_id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
                - `system` (TEXT, UNIQUE): Name of the system.
                - `critical_change_in_system` (BOOLEAN): Flag for system-level critical change.

                ### 3. maintenance_cycle_system (Join Table)
                - **Description:** Links maintenance cycles to systems.
                - **Fields:**
                - `mantention_cycle_id` (INTEGER, FOREIGN KEY → maintenance_cycle)
                - `system_id` (INTEGER, FOREIGN KEY → system)
                - **Primary Key:** Composite (`mantention_cycle_id`, `system_id`)

                ### 4. subsystem
                - **Description:** Contains reference data for subsystems within a system.
                - **Fields:**
                - `subsystem_id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
                - `system_id` (INTEGER, FOREIGN KEY → system)
                - `subsystem` (TEXT): Name of the subsystem.
                - `critical_change_in_subsystem` (BOOLEAN): Flag for subsystem-level critical change.
                - **Unique Constraint:** (`system_id`, `subsystem`)

                ### 5. component
                - **Description:** Contains reference data for components within a subsystem.
                - **Fields:**
                - `component_id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
                - `subsystem_id` (INTEGER, FOREIGN KEY → subsystem)
                - `component` (TEXT): Name of the component.
                - `critical_change_in_component` (BOOLEAN): Flag for component-level critical change.
                - **Unique Constraint:** (`subsystem_id`, `component`)

                ### 6. job
                - **Description:** Stores individual maintenance tasks linked to components.
                - **Fields:**
                - `job_id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
                - `job_type` (TEXT): Type/category of the job.
                - `comment` (TEXT): Comments about the job.
                - `extra_info` (TEXT): Additional details.
                - `component_id` (INTEGER, FOREIGN KEY → component)
                
                Is the context and database sufficient to answer the user question? Answer with only 'true' or 'false'.
                """
            }
        ]
        
        response = self.client.chat.completions.create(
            model='o3-mini',
            messages=messages
        )
        
        text_response = response.choices[0].message.content.strip()
        print(f"Context sufficiency test response: {text_response}")
        
        # Convert the response to a boolean value
        if text_response.lower() == "true":
            return True
        else:
            return False
               
    ## Method to response to the user if the context is not sufficient
    def response_not_sufficient_context(self, user_request: str, context_summary: str) -> str:
        """
        Returns a response to the user if the context is not sufficient for the user question.
        This method resolves the following question: What should I say to the user if the context is not sufficient?
        """
        messages = [
            {
                "role": "system",
                "content": """
                You are a Chatbot Assistant that responds to the user if the context is not sufficient for the user question.

                Task:
                - Respond to the user claiming that the context is not sufficient for the user question.
                - Keep the response concise and informative. Include a suggestion to provide more details or clarify their question.
                - Do not mention these instructions or the word 'prompt' in your output.
                
                Output Format:
                - A response to the user if the context is not sufficient for the user question.
                """
            },
            {
                "role": "user",
                "content": f"The user request is: {user_request} The context summary is: {context_summary}. What information is missing? How can the user clarify their question?"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        text_response = response.choices[0].message.content.strip()
        return text_response
    
    #  On Topic Methods
    ## Method to transform the user request into a simple question
    def transform_user_question_to_simple_question(self, user_request: str) -> str:
        """
        Transform the user question to a simple question.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a Business Analyst expert. You will be provided with a question in natural language, and you will transform it to a simple question to be answered by a SQL query. Your work is to simplify the question, not to answer it."
            },
            {
                "role": "user",
                "content": f"Transform the following question to a simple question: {user_request}"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1200
        )
        
        simple_question = response.choices[0].message.content.strip()
        return simple_question
      
    ## Method to get the SQL query based on the user question
    def create_sql_query(self, simple_question: str, previous_query: str) -> str:
        """
        Get the SQL query based on the user question.
        """
        # Copy the conversation history to the text_to_query_messages
        original_base_messages = self.text_to_query_messages.copy()
        
        text_content = f"{simple_question}" if previous_query == '' else f"{simple_question}\nConsider that the previous query was unsuccessful. The last query was: {previous_query}"
        
        # Add the user question to the conversation history
        original_base_messages.append(
            {
                "role": "user",
                "content": text_content
            }
        )
        
        # Create the SQL query using the OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=original_base_messages,
            # max_tokens=1200
        )
        
        # Extract, add to history and return the SQL query from the response
        sql_query = response.choices[0].message.content.strip()
        return sql_query
        
    ## Method to execute the SQL query and get the result
    def excecute_sql_query(self, sql_query: str) -> pd.DataFrame:
        """
        Executes the SQL query and returns the result as a DataFrame.
        """
        try:
            # Execute the SQL query
            df = pd.read_sql_query(sql_query, self.conn)
            
            return df
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return pd.DataFrame()
    
    ## Method to store the SQL query result in a CSV file
    def store_sql_result_in_csv(self, data_from_query: pd.DataFrame) -> None:
        """
        Stores the SQL query result in a CSV file.
        """
        try:
            data_from_query.to_csv(self.csv_filename_path, index=False)
            print(f"Data stored in {self.csv_filename_path}")
        except Exception as e:
            print(f"Error storing data in CSV: {e}")
        
    ## Method to orchestrate the SQL query execution
    def excecute_sql_process(self, user_request: str, previous_query: str) -> tuple:
        """
        Executes the SQL process needed to answer the user request.
        
        returns:
        - sql_query: str, the SQL query
        - data_from_query: pd.DataFrame, the data from the query
        - is_data_on_dataframe: bool, indicating if the data was retrieved successfully
        """
        # Transform the user request into a simple question
        simple_question = self.transform_user_question_to_simple_question(user_request)
        print(f"✅ Simple question: {simple_question}")
        
        # Create the SQL query needed to answer the user request
        sql_query = self.create_sql_query(simple_question, previous_query)
        print(f"✅ SQL query: {sql_query}")
        
        # Execute the SQL query and get the result
        data_from_query = self.excecute_sql_query(sql_query)
        is_data_on_dataframe = True if len(data_from_query) > 0 else False
        
        if is_data_on_dataframe:
            print('✅ Query was executed successfully\n')
            # Store the SQL query result in a CSV file
            self.store_sql_result_in_csv(data_from_query)
            print('✅ Query result was stored successfully\n')
        
        # Return the SQL query and the data from the query
        return sql_query, data_from_query, is_data_on_dataframe
    
    ## Method to check the sql process is working
    def supervised_sql_process(self, user_request: str, base_query: str = '') -> tuple:
        '''
        Check if the SQL process is working and return the SQL query and the data from the query.
        
        returns:
        - sql_query: str, the SQL query
        - data_from_query: pd.DataFrame, the data from the query
        - is_data_on_dataframe: bool, indicating if the data was retrieved successfully
        '''
        is_data_needed = True
        n_tries = 0
        while is_data_needed and n_tries <= self.max_tries:
            n_tries += 1
            print(f"Attempt {n_tries} to get data from the database.")
            
            # Execute the SQL process to get the data from the database
            sql_query, data_from_query, is_data_on_dataframe = self.excecute_sql_process(user_request, base_query)
            
            if is_data_on_dataframe:
                print("Data was retrieved successfully.")
                is_data_needed = False
            else:
                print("Data was not retrieved successfully. Retrying...")
                base_query = sql_query
                time.sleep(2)
                
        if not is_data_needed:
            # If data was retrieved successfully, return the SQL query and the data from the query
            return sql_query, data_from_query, True
    
        else:
            # If data was not retrieved successfully, return an empty DataFrame
            print("Data was not retrieved successfully after maximum attempts.")
            return '', pd.DataFrame(), False
        
    ## Function to create the final answer using the DataFrame as a string
    def create_final_answer(self, simple_question: str, df: pd.DataFrame) -> str:
        """
        Create the final answer using the question and the DataFrame as a string.
        """
        df_as_string = df.to_string(index=False)
        messages = [
            {
                "role": "system",
                "content": "You are a Business Analyst expert in explaining business and operational questions to your client. You will be provided with the original question of the client and a DataFrame with the answer of the question. Your work consist on answering the question using the provided data in a way that is clear to the client."
            },
            {
                "role": "user",
                "content": f"Client Question: {simple_question}\nAnswer (SQL):{df_as_string}"
            }
        ]
        
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            max_tokens=1200
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
        
    #  Main Methods
    ## Method to Excecute the Agent
    def excecute_agent(self, user_message: str) -> tuple:
        """
        Executes the agent's response based on the conversation history and user question.
        It returns the response to the user and a boolean indicating if the message was able to be answered.
        
        returns:
        - response: str, the response to the user
        - able_to_answer: bool, indicating if the message was able to be answered
        """        
        # Create a summary of the conversation history
        context_summary = self.summarize_context_messages()
        print(f"Context summary: {context_summary}")
        
        # Get the user request based on the conversation history
        user_request = self.get_user_request(user_message, context_summary)
        self.last_user_request = user_request
        print(f"User request: {user_request}")
        
        # Define the nature of the message
        nature = self.get_message_nature(user_request, context_summary)
        print(f"Message nature: {nature}")
        
        # Add the user question to the conversation history
        self.add_message_to_history("user", user_message, nature)
        
        # If the message is off-topic, return a response based on the nature of the message
        if nature != 'on-topic':
            # If the message is not on-topic, return a response based on the nature of the message
            print(f"Message is not on-topic: {nature}. Returning generic response.")
            response = self.excecute_response_not_on_topic(user_message, nature)
            
            # Add the assistant's response to the conversation history
            self.add_message_to_history("assistant", response, nature)
            
            return response, False
        
        # Case when the message is on-topic
        else:
            print("Message is on-topic. Proceeding with context sufficiency check.")
            # Get the context of the conversation and check if it's sufficient for the user question
            context_flag= self.excecute_context_sufficiency_test(user_message, context_summary)
            
            # If the context is not sufficient, proceed with a response asking for more details
            if not context_flag:
                print("Context is not sufficient. Asking the user for more details.")
                response = self.response_not_sufficient_context(user_request, context_summary)
                
                # Add the assistant's response to the conversation history
                self.add_message_to_history("assistant", response, "on-topic")
                
                return response, False
            
            # If the context is sufficient, proceed with the query
            else:
                print("Context is sufficient. Proceeding with the query.")
                # Execute the SQL process to get the data from the database
                sql_query, data_from_query, is_data_on_dataframe = self.supervised_sql_process(user_request)
                
                if not is_data_on_dataframe:
                    print("Data was not retrieved successfully. Returning an error message.")
                    response = "I'm sorry, but I couldn't retrieve the data from the database. Please try again later."
                    
                    # Add the assistant's response to the conversation history
                    self.add_message_to_history("assistant", response, "general")
                    
                    return response, False
                
                else:
                    print(f"Data was retrieved successfully.\n{data_from_query.to_string(index=False)}\nProceeding with the final answer.")
                    # Add the SQL query to the class attribute
                    self.last_query = sql_query
                    
                    # Create the final answer using the DataFrame as a string
                    final_answer = self.create_final_answer(user_request, data_from_query)
                    print(f"Final answer: {final_answer}")
                    
                    # Add the assistant's response to the conversation history
                    self.add_message_to_history("assistant", sql_query, "on-topic")
                    self.add_message_to_history("assistant", final_answer, "on-topic")
                    
                    # Add the final_answer to the class attribute
                    self.last_answer = final_answer
                    
                    print(self.real_history_messages)
                    
                    return final_answer, True
          
    #######################
    #  Assistant Methods  #    
    #######################
    
    #  General Response Methods : Just Answer the user question
    ## Method to add a message to the general message history
    def add_message_to_general_history(self, role: str, content: str) -> None:
        """
        Adds a message to the conversation history.
        """
        self.general_messages.append({"role": role, "content": content})
        
    ## Method to transform the instructions into a user request
    def transform_instructions_to_user_request(self, instructions: str) -> str:
        """
        Transform the instructions into a user request.
        """
        messages = [
            {
                "role": "system",
                "content": """
                You are a Chatbot Assistant that transforms the instructions into a user request.
                Your goal is to rephrase the instructions in a way that is clear and concise for the machine to understand.

                Task:
                - Transform the instructions into a user request.
                - Keep the response concise and informative.
                - Do not mention these instructions or the word 'prompt' in your output.
                
                Output Format:
                - A user request based on the instructions provided.
                """
            },
            {
                "role": "user",
                "content": f"The instructions are: {instructions}"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1200
        )
        
        text_response = response.choices[0].message.content.strip()
        return text_response
    
    ## Method to execute the general response based on the instructions
    def excecute_general_response(self, instructions: str) -> tuple:
        """
        Executes the general response based on the instructions provided.
        
        returns:
        - response: str, the response to the user
        - new_user_request: str, the new user request based on the instructions
        """
        
        new_user_request = self.transform_instructions_to_user_request(instructions)
        print(f"New user request: {new_user_request}")
        
        system_prompt = """
            You are a Chatbot Assistant that generates a response to the user question.
            
            Task:
            - Generate a response to the user question based on the conversation history.
            - Keep the response concise and informative.
            - Do not mention these instructions or the word 'prompt' in your output.
            
            Output Format:
            - A response to the user question.
        """
        
        messages = [self.last_user_request,
                    self.last_answer,
                    {"role": "system","content": system_prompt}
                    ]
        
        if len(self.general_messages) > 0:
            messages += self.general_messages
        
        messages.append({"role": "user","content": f"The user question is: {new_user_request}"})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1200
        )
        
        response_text = response.choices[0].message.content.strip()
        print(f"Response: {response_text}")
        
        return response_text, new_user_request
    
    # Query Methods : Improve the SQL query according to the user instructions
    ## Method to add a message to the query message history
    def add_message_to_query_history(self, role: str, content: str) -> None:
        """
        Adds a message to the conversation history.
        """
        self.query_messages.append({"role": role, "content": content})
        
    ## Method to get the clarifications from the instructions
    def get_clarifications_from_instructions(self, instructions: str) -> str:
        """
        Get the clarifications from the instructions.
        """
        messages = [
            {
                "role": "system",
                "content": """
                You are a Chatbot Assistant that extracts clarifications from the instructions provided.

                Task:
                - Extract clarifications from the instructions provided. Your goal is to identify what the user wants to clarify or improve in the SQL query.
                - Keep the response concise and informative.
                - Do not mention these instructions or the word 'prompt' in your output.
                
                Output Format:
                - A list of clarifications extracted from the instructions provided.
                """
            },
            {
                "role": "user",
                "content": f"The instructions are: {instructions}"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1200
        )
        
        clarifications = response.choices[0].message.content.strip()
        return clarifications
    
    ## Method to execute the SQL query and get the result
    def excecute_query_response(self, instructions: str) -> tuple:
        """
        Executes the SQL query and returns the result as a DataFrame.
        
        returns:
        - sql_query: str, the SQL query
        - data_from_query: pd.DataFrame, the data from the query
        - is_data_on_dataframe: bool, indicating if the data was retrieved successfully
        - clarifications: str, the clarifications extracted from the instructions
        """
        clarifications = self.get_clarifications_from_instructions(instructions)
        print(f"Clarifications: {clarifications}")
    
        sql_query, data_from_query, is_data_on_dataframe = self.supervised_sql_process(clarifications, self.last_query)

        if is_data_on_dataframe:
            print('✅ Query was executed successfully\n')
            self.final_dataframe = data_from_query
            return sql_query, data_from_query, True, clarifications
        
        else:
            print('❌ Query was not executed successfully\n')
            return '', pd.DataFrame(), False, clarifications
            
    #  Image Creation Methods
    ## Method to upload the CSV file to OpenAI and return the file ID
    def upload_file_openai(self) -> str:
        """
        Upload csv file to openai and return the file ID.
        """
        
        file = self.client.files.create(
                file=open(self.csv_filename_path, "rb"),
                purpose='assistants'
                )
        
        return file.id
    
    ## Method to get the instructions for image creation
    def get_instructions_for_image_creation(instructions: str) -> str:
        
        if instructions is None:
            return "return the code as a python script and the image as a png file"
        
        else:
            string_out = f"""
            The previous code to generate the image is:
            {self.final_code}
            
            There are some changes I would like to apply:
            {instructions}
            
            return the improved code as a python script and the improved output image as a png file.
            """
            
            return string_out
            
    ## Method to check the run status and return the messages and the status
    def check_run_status(self, run_id: str, thread_id: str) -> tuple:
        """
        Check the status of the run and return the messages and the status.
        
        returns:
        - messages_data: list, the messages data
        - is_image_created: bool, indicating if the image was created successfully
        """
        is_image_created = False
        unexpected_status = 0
        
        while not is_image_created and unexpected_status < 3:
            # Check the status of the run
            run = self.client.beta.threads.runs.get(run_id)
            status = run.status
            
            if status == 'completed':
                is_image_created = True
                print("Image created successfully.")
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread_id
                )
                break
            elif status == 'failed':
                print("Image creation failed.")
                break
            elif status == 'running':
                print("Image creation is still running...")
                time.sleep(5)
            else:
                print(f"Unexpected status: {status}. Retrying...")
                unexpected_status += 1
                time.sleep(5)
        
        if is_image_created == 'completed':
            return messages.data, is_image_created
        else:
            return None, is_image_created
        
    ## Method to create an image using the assistant
    def create_image(self, instructions: str) -> list:
        """
        Create an image using the assistant.
        """
        
        instructions_text = self.get_instructions_for_image_creation(instructions)
        
        # Create a thread and send the SQL query to the assistant
        thread = self.client.beta.threads.create(
            messages=[
                {
                "role": "user",
                "content": "Write python code to create an intuitive chart with the data and export the image as a png file",
                "attachments": [
                    {
                    "file_id": self.file_id,
                    "tools": [{"type": "code_interpreter"}]
                    }
                ]
                }
            ]
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self.assistant_id,
            instructions=instructions_text,
        )
        
        messages, is_image_created = self.check_run_status(run.id, thread.id)
        
        if is_image_created == 'completed':
            return messages.data
        else:
            return None
        
    ## Method to store the image as a PNG file
    def store_image_as_png(self, image_file_id: str) -> bytes:
        """
        Store the image as a PNG file.
        """
        try:
            # Get the image data from OpenAI
            image_data = self.client.files.content(image_file_id)
            # Read the image data as bytes
            image_data_bytes = image_data.read()
            # Save the image data to a file
            with open(self.chart_filename_path, "wb") as file:
                file.write(image_data_bytes)
                
            print(f"Image stored in {self.chart_filename_path}")
                
            return image_data_bytes
                
        except Exception as e:
            print(f"Error storing image: {e}")
    
    ## Method to filter the code from the text
    def filter_code_in_text(self, full_text: str) -> str:
        """
        Filter the code from the text.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a code expert. You will be provided with a text and you will filter the python code that creates the image. Omit any other text or explanation."
            },
            {
                "role": "user",
                "content": f"Extract the python code from the following text:\n{full_text}"
            }
        ]
        
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            max_tokens=1200
        )
        
        return response.choices[0].message.content.strip()
    
    ## Method to store the code as a Python file
    def store_code_as_py(self, code_text: str) -> str:
        """
        Store the code as a Python file.
        """
        with open(self.code_filename_path, "w") as file:
            file.write(code_text)
        print(f"Code stored in {self.code_filename_path}")
        
        return code_text
    
    ## Method to extract the code and image from the assistant response
    def extract_code_and_image(self, messages: List[dict]) -> tuple:
        """
        Extract the code and image from the messages.
        
        returns:
        - image: bytes, the image data
        - code: str, the code text
        """
        image_message = messages[0]
        code_message = messages[1]
        
        image_file_id = image_message.attachments[0].file_id
        image = self.store_image_as_png(image_file_id)
        print('✅Image was downloaded successfully')
        
        code_text = code_message.content[0].text.value
        code_text = self.filter_code_in_text(code_text)
        print('✅Code was extracted successfully')
        code = self.store_code_as_py(code_text)  
        print('✅Code was stored successfully')
        
        return image, code
    
    ## Method to execute the Python code needed to answer the user request
    def excecute_python_code(self, instructions: str) -> tuple:
        """
        Executes the Python code needed to answer the user request.
        
        returns:
        - image: bytes, the image data
        - code: str, the code text
        """
        # Upload the CSV file to OpenAI and get the file ID
        if self.file_id is None:
            file_id = self.upload_file_openai()
            print(f"✅ File ID:\nFile ID: {file_id}")
            self.file_id = file_id
        
        # Create an image using the assistant
        print('⏳Running the assistant with the uploaded file...')
        messages = self.create_image(instructions)
        print('✅ Assistant finished running')
        
        # Extract the code and image from the assistant response
        image, code = self.extract_code_and_image(messages)
        print('✅ Code and image were extracted successfully')
        
        # Return the image and code
        return image, code
        
    #  Main Method
    ## Method to orchestrate the general assistant response
    def excecute_general_response(self, instructions: str) -> str:
        """
        Executes the general response based on the instructions provided.
        """
        general_response, new_user_message = self.excecute_general_response(instructions)
            
        self.add_message_to_general_history("user", new_user_message)
        self.add_message_to_general_history("assistant", general_response)
        
        return general_response
    
    ## Method to orchestrate the SQL query execution
    def excecute_query_response(self, instructions: str) -> str:
        """
        Executes the SQL query and returns the result as a DataFrame.
        """
        # Execute the SQL query and get the result
        sql_query, data_from_query, is_data_on_dataframe, clarifications = self.excecute_query_response(instructions)
        
        data_based_answer = "I'm sorry, but I couldn't retrieve the data from the database. Please try again later."
        if is_data_on_dataframe:
            # Create the final answer using the DataFrame as a string
            data_based_answer = self.agent.create_final_answer(clarifications, data_from_query)
            
            
            # Add the assistant's response to the conversation history
            self.add_message_to_query_history("user", clarifications)
            self.add_message_to_query_history("assistant", data_based_answer)
            self.last_query = sql_query
            
        print(f"Final answer: {data_based_answer}")
        return data_based_answer
        
    ## Method to orchestrate the Python code execution
    def excecute_image_response(self, instructions: str) -> str:
        # Execute the Python code needed to answer the user request
        print('⏳Running the assistant with the uploaded file...')               
        image, code = self.excecute_python_code(instructions)
        print('✅ Python code was executed successfully')
        
        # Update the class attributes with the image and code
        self.final_image = image
        self.final_code = code
        
        text_response = f"Image created successfully. The image is stored in {self.chart_filename_path} and the code is stored in {self.code_filename_path}."
        return text_response
        
    ## Method to execute the assistant's response based on the user input
    def excecute_assistant(self, user_request: str, instructions: str) -> str:
        """
        Executes the assistant's response based on the user input.
        It returns the response to the user and a boolean indicating if the message was able to be answered.
        
        returns:
        - response: str, the response to the user according to the user request
        """
        if user_request == 'General':
            # Execute the general response based on the instructions
            print('⏳Running the assistant with the uploaded file...')
            general_response = self.excecute_general_response(instructions)
            print('✅ Assistant finished running')
            
            return general_response
            
        elif user_request == 'Query':
            # Execute the SQL query and get the result
            print('⏳Running the assistant with the uploaded file...')
            query_response = self.excecute_query_response(instructions)
            print('✅ Assistant finished running')
            
            return query_response
            
            
        elif user_request == 'Image':
            # Execute the Python code needed to answer the user request
            print('⏳Running the assistant with the uploaded file...')
            image_response = self.excecute_image_response(instructions)
            print('✅ Assistant finished running')
            
            return image_response
        
        else:
            print("Invalid user request. Please provide a valid request.")
            return None    