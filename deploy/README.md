# Hosting on a Cloudflare domain

This guide puts the dashboard on a domain you manage in Cloudflare, for example
`https://fire.example.com/dashboard/`. Replace `fire.example.com` with your own
hostname throughout.

Two containers run on your server:

- **app** serves the API and the dashboard. It listens only on the server's
  loopback address, so it is never reachable directly from the internet.
- **tunnel** runs `cloudflared`. It opens an outbound connection to Cloudflare,
  and Cloudflare forwards visitors to the app through it. You do not open any
  inbound firewall ports or install TLS certificates; Cloudflare serves HTTPS.

The server only serves forecasts. Training stays on your own computer, and you
upload new forecast files when you have them.

## 1. Before you start

- **A Linux server** that stays on, such as a small VPS, with Docker Engine and
  the Docker Compose plugin and Python 3.12 or newer installed (for the update
  helper). The image is tested on x86-64. Allow at
  least 1 GB of RAM and 2 GB of free disk. The running app uses about 250 MB.
- **Your domain on Cloudflare.** The domain's status in the Cloudflare dashboard
  must be *Active*, which means its nameservers point to Cloudflare.
- **A NASA FIRMS MAP_KEY** for the live alert feed. Forecasts still display
  without it.
- **Forecast files on your computer.** Run `python src/training.py` first if
  `outputs/` is empty.

Make Docker start at boot so the site returns after a server restart:

```bash
sudo systemctl enable --now docker
```

## 2. Package the project on your computer

From the project root on your computer:

```bash
release_file="$HOME/wildfire-$(date +%Y%m%d-%H%M%S).tar.gz"
python deploy/package.py --output "$release_file"
scp "$release_file" user@your-server:~/
basename "$release_file"       # use this exact filename on the server
```

The archive contains the code, the deployment files and the forecast outputs.
It excludes `.env`, `.cloudflare.env`, training caches and build folders, and
rejects symbolic links, hard links and special files. Keep credentials in the
excluded environment files. A failed package is removed; an existing archive is
never overwritten. The filename includes the time to distinguish releases.

## 3. Start the app on the server

On the server, replace `YYYYMMDD-HHMMSS` with the exact filename printed above.
Do not use `*`: older archives may also be present.

```bash
release_file="$HOME/wildfire-YYYYMMDD-HHMMSS.tar.gz"
tar -xzf "$release_file" -C "$HOME"
cd wildfire
cp .env.example .env
nano .env                      # set FIRMS_API_KEY=your MAP_KEY
docker compose up -d --build
docker compose ps              # wait until app shows "healthy"
curl -fsS http://127.0.0.1:8000/ && echo
```

The last command should print `{"status":"running",...}`. If `docker compose ps`
never shows `healthy`, see [Troubleshooting](#troubleshooting).

The data paths in `.env.example` point at your local folders. You do not need
to change them; `compose.yaml` overrides them with the container's paths.

## 4. Create the Cloudflare Tunnel

In the [Cloudflare dashboard](https://dash.cloudflare.com/):

1. Go to **Networking** > **Tunnels** and select **Create Tunnel**.
2. Enter a name such as `wildfire` and select **Create Tunnel**.
3. Under **Setup Environment**, pick **Docker** if it is listed. Any choice
   works, because you only need the token from the install command. Copy only
   the long string after `--token`; it starts with `eyJ`. Do not run the
   command itself.

   ```text
   docker run cloudflare/cloudflared:latest tunnel --no-autoupdate run --token eyJhIjoi...
   ```

4. Leave the page open and switch to the server.

On the server, store the token and start the tunnel next to the app:

```bash
cp deploy/cloudflare.env.example .cloudflare.env
chmod 600 .cloudflare.env
nano .cloudflare.env           # TUNNEL_TOKEN=eyJhIjoi...
docker compose -f compose.yaml -f compose.cloudflare.yaml up -d
docker compose -f compose.yaml -f compose.cloudflare.yaml logs tunnel
```

The log should show `Registered tunnel connection` several times. Back in the
dashboard, the tunnel status changes to **Healthy**. Select **Continue**.

## 5. Publish the hostname

1. Open the tunnel, then under **Routes** select **Add route** >
   **Published application**.
2. For **Hostname**, enter a subdomain such as `fire` and pick your domain.
   Leave the path empty.
3. For **Service URL**, enter exactly:

   ```text
   http://app:8000
   ```

   Use `app`, the Compose service name. Inside the tunnel container,
   `localhost` refers to the tunnel itself, so `http://localhost:8000` returns
   a 502 error.
4. Select **Add route**.
5. In your domain's **DNS** > **Records**, confirm there is a proxied `CNAME`
   for `fire` pointing to `<tunnel-id>.cfargotunnel.com`. Cloudflare normally
   creates it; add it yourself if it is missing.

Open `https://fire.example.com/dashboard/`. The first load can take a minute
while DNS propagates.

## 6. Send the bare domain to the dashboard

The site root `https://fire.example.com/` returns the API's JSON status, which
API clients use. To send people who type the bare hostname to the dashboard,
add a Cloudflare redirect rule instead of changing the API:

1. In your domain, go to **Rules** and create a **Redirect Rule**.
2. Set it to match requests where **Hostname** equals `fire.example.com`
   **and** **URI Path** equals `/`.
3. Redirect to the static URL `https://fire.example.com/dashboard/` with status
   code `302`, and deploy the rule.

The API documentation stays at `https://fire.example.com/docs`.

## 7. Update the forecasts

After you retrain on your computer, package and upload a new release using
step 2. Wait for `scp` to finish. Then on the server, use its exact filename:

```bash
cd ~/wildfire
python3 deploy/update_forecasts.py "$HOME/wildfire-YYYYMMDD-HHMMSS.tar.gz"
```

The helper extracts outputs into a separate staging folder and validates the
JSON, latest snapshot and compressed observation file while the current app
continues serving. An incomplete or invalid archive is rejected before stopping
the app. Files receive read permissions for the container's non-root user.

Activation causes a brief interruption: the helper stops the app, moves the old
outputs to `.forecast-backups/`, installs the staged directory, and recreates the
app container so Docker mounts the new directory. It waits for the health check;
if startup fails, it restores the previous outputs and recreates the app again.
The tunnel stays running throughout. Do not extract or stream files directly
over the running app's `outputs/` directory.

The helper installs only forecasts; it leaves code and credentials in place.
Backups are retained for recovery. Allow room for the current outputs, the new
release and retained backups; remove older backups after verifying the update.
Run one update at a time. Use the default Compose project name `wildfire` from
`compose.yaml`, as in this guide.

To ship code changes as well, upload a complete archive as in step 2. Schedule
a maintenance interruption: stop the app before extracting any replacement
files, then rebuild and start it. Keep the previous release archive for recovery.
Extracting keeps `.env` and `.cloudflare.env`, which the archive excludes.

```bash
cd ~/wildfire
docker compose -f compose.yaml -f compose.cloudflare.yaml stop app
tar -xzf "$HOME/wildfire-YYYYMMDD-HHMMSS.tar.gz" -C "$HOME"
docker compose -f compose.yaml -f compose.cloudflare.yaml up -d --build --wait
```

## Everyday commands

Run these from the `wildfire` folder on the server. An alias saves typing:

```bash
alias wf='docker compose -f compose.yaml -f compose.cloudflare.yaml'
wf ps                          # status of both containers
wf logs -f app                 # follow the app's log
wf up -d --force-recreate app  # apply changes made to .env
wf down                        # take the site offline
```

`docker compose restart` does not re-read `.env`, so use `--force-recreate`
after editing it.

`cloudflared` is pinned to a tested release in `compose.cloudflare.yaml`.
Cloudflare supports each release for about a year, so every few months change
the `image:` tag to the newest version listed on
[Docker Hub](https://hub.docker.com/r/cloudflare/cloudflared/tags) and run
`wf up -d`.

## Security

### Already enforced by the project

- **Network.** The app port is bound to `127.0.0.1`, so the only public way in
  is the tunnel. Do not change that to `0.0.0.0` or open port 8000.
- **Containers.** Both run with a read-only filesystem, no Linux capabilities,
  `no-new-privileges`, memory and process limits, and rotated logs. The app runs
  as a non-root user and accepts at most 100 connections at once.
- **Browser protection.** Every response carries security headers, including a
  Content Security Policy that lets the dashboard load code only from this site
  and the two pinned CDNs. Each CDN file is pinned by an integrity hash, so a
  modified copy is refused. The headers live in `src/security.py`.
- **Secrets.** `.env` and `.cloudflare.env` are excluded from Git, the Docker
  image and the deployment archive.

### Turn on in Cloudflare

In your domain in the [Cloudflare dashboard](https://dash.cloudflare.com/):

1. **SSL/TLS** > **Edge Certificates**: turn on **Always Use HTTPS**, set
   **Minimum TLS Version** to TLS 1.2, and enable **HSTS** with a 6-month
   max-age. Leave *include subdomains* and *preload* off unless every subdomain
   of your domain serves HTTPS.
2. **Security rules** > **Create rule** > **Rate limiting rules**. The free plan
   allows one rule:
   - Match requests where **Hostname** equals `fire.example.com`.
   - Count by **IP**. When the rate exceeds **100 requests per 10 seconds**,
     **Block** for **10 seconds**.

   A dashboard visit makes about 20 requests, so a single visitor stays far
   below the limit, while scripted floods are cut off before they reach your
   server. Visitors on one shared network, such as a classroom, count as one
   IP address. Raise the limit or pause the rule during a group demonstration.
3. Leave the free **Managed rules** on. Cloudflare enables them by default.
4. Optional: to limit the site to specific people, add a Cloudflare Access
   application for the hostname. The app itself has no login.

### Harden the server

On a Debian or Ubuntu server:

```bash
sudo ufw default deny incoming
sudo ufw allow OpenSSH
sudo ufw enable
sudo apt install unattended-upgrades
```

The tunnel only makes outbound connections, so SSH is the only inbound port
you need. Use SSH keys and set `PasswordAuthentication no` in
`/etc/ssh/sshd_config`.

### Keep it patched

- **Monthly**, rebuild on the newest Python base image:
  `wf build --pull && wf up -d`.
- **Before each release**, check the pinned Python packages for known
  vulnerabilities on your computer:

  ```bash
  python -m pip install pip-audit
  pip-audit -r deploy/requirements.lock
  ```

- **If the tunnel token leaks**, open the tunnel in the dashboard, refresh its
  token, and paste the new one into `.cloudflare.env`.
- **When the dashboard gains a CDN library or calls a new service**, add the
  library's `integrity` hash in `frontend/index.html` and the new origin in
  `src/security.py`. `tests/test_security.py` fails until both are present.
  Compute a hash with:

  ```bash
  curl -fsSL https://cdn.example/lib.js | openssl dgst -sha384 -binary | openssl base64 -A
  ```

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `app` stays `unhealthy` and its log shows `PermissionError` | Files were copied without read permission for other users. Run `chmod -R a+rX data outputs` in the `wildfire` folder. |
| `app` log says `Run python src/training.py ...` | `outputs/` is missing forecast files. Package again from a computer that has them. |
| Cloudflare error 1033 | The tunnel is not connected. Check `wf logs tunnel` and the token in `.cloudflare.env`. Outbound port 7844 must be allowed. |
| `tunnel` log repeats `Provided Tunnel token is not valid` | `.cloudflare.env` still holds the placeholder, or the token was cut off when pasted. Paste the whole token again, then run `wf up -d`. |
| Cloudflare 502 Bad Gateway | The route's service URL is wrong. It must be `http://app:8000`. |
| `tunnel` exits with `depends_on ... unhealthy` | The tunnel waits for the app's health check. Fix the app first. |
| Live alerts say the MAP_KEY is missing | Set `FIRMS_API_KEY` in `.env`, then run `wf up -d --force-recreate app`. |
| A dashboard feature stops loading after a frontend change | The browser console shows `Refused to load ... Content Security Policy`. Add the new origin to `src/security.py`, or the missing `integrity` hash to `frontend/index.html`. |
| `port is already allocated` when starting | Another program uses port 8000, such as a local development server. Add `WEB_PORT=8001` to `.env` and use that port in the `curl` check. The tunnel is unaffected because it reaches the app inside Docker. |
