AI Driver - a PolyModLoader mod for PolyTrack 0.6.3

OPTION A - nothing to install (browser)
1. Put this folder online with free GitHub Pages:
   - make a free GitHub account, create a public repository (for example "polytrack-mods")
   - "Add file" > "Upload files", drag the whole "polytrack-ai-driver" folder in, commit
   - Settings > Pages > Deploy from a branch > main / root > Save, then wait about a minute
   - check that  https://YOURNAME.github.io/polytrack-mods/polytrack-ai-driver/manifest.json  shows some text
2. Open PolyModLoader's website (the link on its Nexus Mods page), open the mods menu and add a mod by URL:
       https://YOURNAME.github.io/polytrack-mods/polytrack-ai-driver

OPTION B - run it from your own computer (needs Node.js)
1. In the folder that CONTAINS "polytrack-ai-driver", run:   npx http-server -c-1 --cors
2. Add this URL in PolyModLoader's mods menu:   http://localhost:8080/polytrack-ai-driver

USE
- Open a track and press K. The AI restarts the track and starts trying. Press K again to stop.
- Shift+K forgets what it learned on the current track.
- If something goes wrong, press F12 and look for console lines starting with [AI Driver].

NOTES
- Needs PolyTrack 0.6.3. On another version it shows an alert and does nothing.
- AI-driven runs cannot be uploaded to the leaderboard, and the mod marks the game as modded for multiplayer.
- The AI uses the game's default keys for restart (T) and accelerate (W). Edit CFG at the top of main.js if you rebind them.
