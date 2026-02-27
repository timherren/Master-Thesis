# Guided AI Agent Experience

## Overview

The chatbot now provides **complete guidance** at every step. Users never have to guess what they can do - the agent proactively shows all available options with clickable action buttons.

## Key Features

### 1. **Action Suggestions at Every Step**

Every agent response includes:
- **Clear action buttons** showing what the user can do next
- **Descriptions** explaining each action
- **Example commands** users can type or click

### 2. **Structured Guidance**

The agent provides:
- **Current status** - Where you are in the workflow
- **Available actions** - What you can do right now
- **Next steps** - Recommended path forward
- **Alternative options** - Other things you can try

### 3. **Clickable Action Buttons**

The UI automatically converts action suggestions into clickable buttons:
- Click a button → Command is sent automatically
- No need to type commands manually
- Visual feedback on available options

## Workflow Steps with Guidance

### Step 1: Initial (No Data)
**Agent shows:**
- Upload data button
- Ask a causal question option
- Clear instructions on how to start

**Actions:**
- 📁 Upload your data
- 💬 Ask a causal question

---

### Step 2: DAG Proposed
**Agent shows:**
- Proposed DAG structure (variables and edges)
- Clear next steps

**Actions:**
- ✅ Test this DAG against your data
- ✏️ Modify the DAG structure (add/remove edges)
- 🔄 Ask for a different DAG

**Example commands:**
- `yes` - Test the DAG
- `add x1 -> x3` - Add an edge
- `no, propose a different DAG` - Get alternative

---

### Step 3: DAG Tested
**Agent shows:**
- Test results (consistent/inconsistent)
- Number of rejected CI tests
- Interpretation of results

**If consistent:**
- ✅ The DAG is consistent with your data!
- 🚀 Fit the TRAM-DAG model
- ✏️ Modify the DAG anyway

**If inconsistent:**
- ⚠️ Some assumptions were rejected
- 🔧 Get revision suggestions
- ✏️ Manually modify the DAG
- ✅ Proceed anyway (not recommended)

---

### Step 4: DAG Revisions (if needed)
**Agent shows:**
- LLM-suggested revisions
- Explanation of why revisions are needed

**Actions:**
- ✏️ Apply these revisions
- ✏️ Manually modify the DAG
- 🚀 Proceed to model fitting
- 🔄 Test the DAG again

---

### Step 5: Model Fitted
**Agent shows:**
- Training summary
- Loss history availability
- Experiment directory

**Actions:**
- 📊 Compute Average Treatment Effect (ATE)
- 🔬 Perform intervention analysis
- 📈 Generate plots and visualizations
- 📋 Show associations
- 📄 Download full report

**Example commands:**
- `What is the effect of x1 on x3?` - Compute ATE
- `What if x1 = 3?` - Intervention analysis
- `sample 1000 data points and show me the plots` - Generate plots
- `show associations` - Correlation matrix
- `download report` - Get PDF report

---

### Step 6: Query Answering
**After ATE computation:**
- 📄 Download intervention report
- 🔬 Try different intervention values
- 📊 Compute ATE for other variables
- 📈 Generate plots

**After plot generation:**
- 📄 Download full PDF report
- 🔬 Ask more causal questions
- 📊 Generate more plots

**After association analysis:**
- 🔬 Compute causal effects (ATE)
- 📈 Generate plots
- 📄 Download full report

---

## UI Features

### Action Buttons
- **Visual design**: Blue background, clear labels
- **Hover effects**: Buttons highlight on hover
- **Click to execute**: One click sends the command
- **Descriptions**: Each button shows what it does

### Message Formatting
- **Bold headers**: Important information stands out
- **Code blocks**: Commands shown in monospace
- **Lists**: Structured information
- **Emojis**: Visual indicators for different action types

### Status Indicators
- **Step indicators**: Show current workflow position
- **Progress indicators**: Animated dots during processing
- **Status bar**: Shows connection and processing status

## Example Interaction Flow

1. **User opens chatbot**
   - Sees welcome message with action buttons
   - Clicks "Upload your data" or types a question

2. **After data upload**
   - Agent asks: "What would you like to investigate?"
   - User types: "What is the effect of treatment on outcome?"

3. **DAG proposed**
   - Agent shows proposed DAG
   - **Action buttons appear:**
     - ✅ Test this DAG
     - ✏️ Modify the DAG
     - 🔄 Ask for different DAG
   - User clicks "Test this DAG"

4. **DAG tested**
   - Agent shows test results
   - **Action buttons appear:**
     - 🚀 Fit the TRAM-DAG model (if consistent)
     - 🔧 Get revision suggestions (if inconsistent)
   - User clicks "Fit the TRAM-DAG model"

5. **Model fitted**
   - Agent shows training summary
   - **Action buttons appear:**
     - 📊 Compute ATE
     - 🔬 Perform intervention
     - 📈 Generate plots
     - 📄 Download report
   - User clicks "Compute ATE"

6. **ATE computed**
   - Agent shows results
   - **Action buttons appear:**
     - 📄 Download report
     - 🔬 Try different values
     - 📊 Compute for other variables
   - User clicks "Download report"

## Benefits

✅ **No guessing** - Users always know what they can do  
✅ **Clear guidance** - Step-by-step instructions  
✅ **Visual feedback** - Action buttons show options  
✅ **Flexible** - Can type commands or click buttons  
✅ **Educational** - Descriptions explain each action  
✅ **Progressive** - Guidance adapts to workflow state  

## Implementation Details

### Backend (`chatbot_server.py`)
- `_format_action_suggestions()` - Formats actions consistently
- Every step method includes action suggestions
- Actions are context-aware (use actual variable names)

### Frontend (`chatbot_ui.html`)
- `addMessage()` - Parses action suggestions
- `sendAction()` - Handles button clicks
- Action buttons styled with Tailwind CSS
- Automatic command execution on click

## Action Format

Actions are structured as:
```python
{
    "label": "📊 Compute ATE",
    "command": "What is the effect of x1 on x3?",
    "description": "Get the causal effect between two variables"
}
```

Displayed as:
- **Label** (with emoji)
- **Description** (what it does)
- **Command** (what will be sent)

---

**Result**: Users have complete visibility into what they can do at every step, with both visual buttons and natural language options. No guessing required! 🎯
