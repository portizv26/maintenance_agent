text_to_query_system = f"""
You are a SQL expert. You will be provided with a question in natural language, and you will generate a SQL query to answer that question.

The structure follows the schema of the database:

                     +--------------------------+
                     |   maintenance_cycle      |
                     |--------------------------|
                     | mantention_cycle_id (PK) |
                     | UnitId                   |
                     | start_time               |
                     | end_time                 |
                     | is_scheduled             |
                     | has_critical_change      |
                     | extra_comments           |
                     +-------------+------------+
                                   |
                                   |  M:N (via maintenance_cycle_system)
                                   v
                     +-------------------------------+
                     | maintenance_cycle_system      |
                     |-------------------------------|
                     | mantention_cycle_id (FK,PK)   |
                     | system_id          (FK,PK)    |
                     +-------------+-----------------+
                                   |
                                   v
                     +--------------------------+
                     |        system            |
                     |--------------------------|
                     | system_id (PK)           |
                     | system (name)            |
                     | critical_change_in_system|
                     +-------------+------------+
                                   |
                                   | 1 : N
                                   v
                     +--------------------------+
                     |       subsystem          |
                     |--------------------------|
                     | subsystem_id (PK)        |
                     | system_id (FK)           |
                     | subsystem (name)         |
                     | critical_change_in_subsystem |
                     +-------------+------------+
                                   |
                                   | 1 : N
                                   v
                     +--------------------------+
                     |       component          |
                     |--------------------------|
                     | component_id (PK)        |
                     | subsystem_id (FK)        |
                     | component (name)         |
                     | critical_change_in_component |
                     +-------------+------------+
                                   |
                                   | 1 : N
                                   v
                     +--------------------------+
                     |          job             |
                     |--------------------------|
                     | job_id (PK)              |
                     | job_type                 |
                     | comment                  |
                     | extra_info               |
                     | component_id (FK)        |
                     +--------------------------+

# Maintenance Database Schema Documentation

## Overview

The **MaintenanceDB** is a SQLite database designed to manage detailed maintenance data for industrial equipment. It captures both the operational records (maintenance cycles and jobs) and the reference hierarchy (systems, subsystems, and components) that defines the pre-established relationships for the equipment.

## Data Stored

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

You will be provided with a question in natural language, and you will generate a SQL query to answer that question.
The answer should only contain the text on SQL query, without any additional text, explanation or characters.
If the query asks for downtime, you should return it on minutes.

Output Format:
SQL Query : string

"""

# Maintenance Cycles per Unit - User Example
text_to_query_user_example = """
I want to know how many maintenance cycles have been performed.
"""

# Maintenance Cycles per Unit - Assistant Example
text_to_query_assistant_example = """
SELECT UnitId, COUNT(*) AS maintenance_cycle_count FROM maintenance_cycle GROUP BY UnitId;
"""

# History of messages
text_to_query_messages = [
    {"role": "system", "content": text_to_query_system},
    {"role": "user", "content": text_to_query_user_example},
    {"role": "assistant", "content": text_to_query_assistant_example},
]

assistant_id = 'asst_feuOGXexFv0jk8EVDs4kR50E'

ground_truth = {
    "What is the total number of maintenance cycles recorded?": 
        "SELECT COUNT(*) AS total_cycles FROM maintenance_cycle",
    
    "How many maintenance cycles are scheduled versus unscheduled?": 
        "SELECT is_scheduled, COUNT(*) AS count FROM maintenance_cycle GROUP BY is_scheduled",
    
    "What is the overall distribution of maintenance cycles by critical versus non-critical changes?": 
        "SELECT has_critical_change, COUNT(*) AS count FROM maintenance_cycle GROUP BY has_critical_change",
    
    "How many maintenance cycles occurred per month/quarter/year?": 
        "SELECT strftime('%Y-%m', start_time) AS period, COUNT(*) AS count FROM maintenance_cycle GROUP BY period ORDER BY period",
    
    "What is the average downtime per maintenance cycle?": 
        "SELECT AVG((julianday(end_time) - julianday(start_time)) * 24 * 60) AS avg_downtime_minutes FROM maintenance_cycle WHERE end_time IS NOT NULL",
    
    "What percentage of maintenance cycles involve at least one critical change?": 
        "SELECT (SUM(CASE WHEN has_critical_change = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS percentage FROM maintenance_cycle",
    
    "How has the number of maintenance cycles changed over time (trend analysis)?": 
        "SELECT strftime('%Y-%m', start_time) AS period, COUNT(*) AS cycles FROM maintenance_cycle GROUP BY period ORDER BY period",
    
    "What is the distribution of maintenance cycles by UnitId (or equipment type) over different time periods?": 
        "SELECT UnitId, COUNT(*) AS cycles FROM maintenance_cycle GROUP BY UnitId ORDER BY cycles DESC",
    
    "Which systems have the highest maintenance frequency in recent periods?": 
        "SELECT s.system, COUNT(mcs.mantention_cycle_id) AS cycles FROM system s JOIN maintenance_cycle_system mcs ON s.system_id = mcs.system_id GROUP BY s.system ORDER BY cycles DESC",
    
    "What is the average downtime for scheduled versus non-scheduled maintenance cycles?": 
        "SELECT is_scheduled, AVG((julianday(end_time) - julianday(start_time)) * 24 * 60) AS avg_downtime FROM maintenance_cycle GROUP BY is_scheduled",
    
    "Which system appears most frequently in maintenance cycles?": 
        "SELECT s.system, COUNT(mcs.mantention_cycle_id) AS cycles FROM system s JOIN maintenance_cycle_system mcs ON s.system_id = mcs.system_id GROUP BY s.system ORDER BY cycles DESC LIMIT 1",
    
    "Which subsystem and component record the highest number of jobs?": 
        """SELECT sub.subsystem, comp.component, COUNT(job.job_id) AS job_count 
           FROM job 
           JOIN component comp ON job.component_id = comp.component_id 
           JOIN subsystem sub ON comp.subsystem_id = sub.subsystem_id 
           GROUP BY sub.subsystem, comp.component 
           ORDER BY job_count DESC LIMIT 1""",
    
    "What are the most common job types across the workshop?": 
        "SELECT job_type, COUNT(*) AS count FROM job GROUP BY job_type ORDER BY count DESC",
    
    "What is the mean downtime for maintenance cycles for systems flagged with critical changes?": 
        """SELECT AVG((julianday(mc.end_time) - julianday(mc.start_time)) * 24 * 60) AS avg_downtime 
           FROM maintenance_cycle mc 
           WHERE mc.mantention_cycle_id IN (
               SELECT mcs.mantention_cycle_id 
               FROM maintenance_cycle_system mcs 
               JOIN system s ON mcs.system_id = s.system_id 
               WHERE s.critical_change_in_system = 1
           )""",
    
    "How many jobs are performed per maintenance cycle on average?": 
        """SELECT AVG(job_count) AS avg_jobs_per_cycle FROM (
               SELECT mc.mantention_cycle_id, COUNT(job.job_id) AS job_count
               FROM maintenance_cycle mc
               JOIN maintenance_cycle_system mcs ON mc.mantention_cycle_id = mcs.mantention_cycle_id
               JOIN system s ON mcs.system_id = s.system_id
               JOIN subsystem sub ON s.system_id = sub.system_id
               JOIN component comp ON sub.subsystem_id = comp.subsystem_id
               JOIN job ON comp.component_id = job.component_id
               GROUP BY mc.mantention_cycle_id
           ) t""",
    
    "What is the total number of maintenance cycles for the 'Motor' system?": 
        """SELECT COUNT(DISTINCT mcs.mantention_cycle_id) AS cycles_count 
           FROM maintenance_cycle_system mcs 
           JOIN system s ON mcs.system_id = s.system_id 
           WHERE s.system = 'Motor'""",
    
    "What is the total number of maintenance cycles for the 'Transmision' system?": 
        """SELECT COUNT(DISTINCT mcs.mantention_cycle_id) AS cycles_count 
           FROM maintenance_cycle_system mcs 
           JOIN system s ON mcs.system_id = s.system_id 
           WHERE s.system = 'Transmision'""",
    
    "What is the average downtime for maintenance cycles involving the 'Motor' system?": 
        """SELECT AVG((julianday(mc.end_time) - julianday(mc.start_time)) * 24 * 60) AS avg_downtime 
           FROM maintenance_cycle mc 
           WHERE mc.mantention_cycle_id IN (
               SELECT mcs.mantention_cycle_id 
               FROM maintenance_cycle_system mcs 
               JOIN system s ON mcs.system_id = s.system_id 
               WHERE s.system = 'Motor'
           )""",
    
    "What is the average downtime for maintenance cycles involving the 'Transmision' system?": 
        """SELECT AVG((julianday(mc.end_time) - julianday(mc.start_time)) * 24 * 60) AS avg_downtime 
           FROM maintenance_cycle mc 
           WHERE mc.mantention_cycle_id IN (
               SELECT mcs.mantention_cycle_id 
               FROM maintenance_cycle_system mcs 
               JOIN system s ON mcs.system_id = s.system_id 
               WHERE s.system = 'Transmision'
           )""",
    
    "Which job types are most common for maintenance tasks on the 'Motor' system?": 
        """SELECT job.job_type, COUNT(job.job_id) AS count 
           FROM maintenance_cycle_system mcs 
           JOIN system s ON mcs.system_id = s.system_id 
           JOIN subsystem sub ON s.system_id = sub.system_id 
           JOIN component comp ON sub.subsystem_id = comp.subsystem_id 
           JOIN job ON comp.component_id = job.component_id 
           WHERE s.system = 'Motor'
           GROUP BY job.job_type ORDER BY count DESC""",
    
    "Which job types are most common for maintenance tasks on the 'Transmision' system?": 
        """SELECT job.job_type, COUNT(job.job_id) AS count 
           FROM maintenance_cycle_system mcs 
           JOIN system s ON mcs.system_id = s.system_id 
           JOIN subsystem sub ON s.system_id = sub.system_id 
           JOIN component comp ON sub.subsystem_id = comp.subsystem_id 
           JOIN job ON comp.component_id = job.component_id 
           WHERE s.system = 'Transmision'
           GROUP BY job.job_type ORDER BY count DESC""",
    
    "How do scheduled and unscheduled maintenance cycles compare between 'Motor' and 'Transmision' systems?": 
        """SELECT s.system, mc.is_scheduled, COUNT(DISTINCT mc.mantention_cycle_id) AS cycles 
           FROM maintenance_cycle mc 
           JOIN maintenance_cycle_system mcs ON mc.mantention_cycle_id = mcs.mantention_cycle_id 
           JOIN system s ON mcs.system_id = s.system_id 
           WHERE s.system IN ('Motor', 'Transmision')
           GROUP BY s.system, mc.is_scheduled""",
    
    "What proportion of maintenance cycles in 'Motor' and 'Transmision' involve critical changes?": 
        """SELECT s.system, (SUM(CASE WHEN mc.has_critical_change = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(DISTINCT mc.mantention_cycle_id)) AS percentage 
           FROM maintenance_cycle mc 
           JOIN maintenance_cycle_system mcs ON mc.mantention_cycle_id = mcs.mantention_cycle_id 
           JOIN system s ON mcs.system_id = s.system_id 
           WHERE s.system IN ('Motor', 'Transmision')
           GROUP BY s.system""",
    
    "Which UnitIds (equipment units) have the highest frequency of maintenance cycles?": 
        "SELECT UnitId, COUNT(*) AS cycles FROM maintenance_cycle GROUP BY UnitId ORDER BY cycles DESC",
    
    "What is the average number of jobs per maintenance cycle by system?": 
        """SELECT system, AVG(job_count) AS avg_jobs FROM (
               SELECT s.system, mc.mantention_cycle_id, COUNT(job.job_id) AS job_count
               FROM maintenance_cycle mc
               JOIN maintenance_cycle_system mcs ON mc.mantention_cycle_id = mcs.mantention_cycle_id
               JOIN system s ON mcs.system_id = s.system_id
               JOIN subsystem sub ON s.system_id = sub.system_id
               JOIN component comp ON sub.subsystem_id = comp.subsystem_id
               JOIN job ON comp.component_id = job.component_id
               GROUP BY mc.mantention_cycle_id, s.system
           ) t GROUP BY system""",
    
    "How many maintenance cycles include multiple systems in the same cycle?": 
        """SELECT COUNT(*) AS cycles_with_multiple_systems 
           FROM (
               SELECT mantention_cycle_id, COUNT(system_id) AS system_count 
               FROM maintenance_cycle_system GROUP BY mantention_cycle_id 
               HAVING system_count > 1
           ) t""",
    
    "Are there patterns in the occurrence of maintenance cycles (e.g., seasonal effects)?": 
        "SELECT strftime('%m', start_time) AS month, COUNT(*) AS cycles FROM maintenance_cycle GROUP BY month ORDER BY month",
    
    "What is the average response time for maintenance cycles per subsystem?": 
        """SELECT sub.subsystem, AVG((julianday(mc.end_time) - julianday(mc.start_time)) * 24 * 60) AS avg_response_time 
           FROM maintenance_cycle mc 
           JOIN maintenance_cycle_system mcs ON mc.mantention_cycle_id = mcs.mantention_cycle_id 
           JOIN system s ON mcs.system_id = s.system_id 
           JOIN subsystem sub ON s.system_id = sub.system_id 
           WHERE mc.end_time IS NOT NULL 
           GROUP BY sub.subsystem""",
    
    "How does the downtime for components with critical changes compare to those without?": 
        """SELECT 'Critical Downtime' AS type, AVG((julianday(mc.end_time) - julianday(mc.start_time)) * 24 * 60) AS avg_downtime 
           FROM maintenance_cycle mc 
           WHERE mc.mantention_cycle_id IN (
               SELECT DISTINCT mc2.mantention_cycle_id 
               FROM maintenance_cycle mc2 
               JOIN maintenance_cycle_system mcs ON mc2.mantention_cycle_id = mcs.mantention_cycle_id 
               JOIN system s ON mcs.system_id = s.system_id 
               JOIN subsystem sub ON s.system_id = sub.system_id 
               JOIN component comp ON sub.subsystem_id = comp.subsystem_id 
               WHERE comp.critical_change_in_component = 1
           )
           UNION ALL
           SELECT 'Non-Critical Downtime', AVG((julianday(mc.end_time) - julianday(mc.start_time)) * 24 * 60)
           FROM maintenance_cycle mc 
           WHERE mc.mantention_cycle_id NOT IN (
               SELECT DISTINCT mc2.mantention_cycle_id 
               FROM maintenance_cycle mc2 
               JOIN maintenance_cycle_system mcs ON mc2.mantention_cycle_id = mcs.mantention_cycle_id 
               JOIN system s ON mcs.system_id = s.system_id 
               JOIN subsystem sub ON s.system_id = sub.system_id 
               JOIN component comp ON sub.subsystem_id = comp.subsystem_id 
               WHERE comp.critical_change_in_component = 1
           )""",
    
    "What are the most recurring issues (as indicated by job comments) in maintenance records?": 
        "SELECT comment, COUNT(*) AS count FROM job GROUP BY comment ORDER BY count DESC LIMIT 5"
}
