## GitHub Copilot Chat

- Extension Version: 0.31.0 (prod)
- VS Code: vscode/1.104.0
- OS: Windows

## Network

User Settings:
```json
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 20.207.73.85 (58 ms)
- DNS ipv6 Lookup: Error (57 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy URL: None (15 ms)
- Electron fetch (configured): HTTP 200 (1087 ms)
- Node.js https: HTTP 200 (94 ms)
- Node.js fetch: HTTP 200 (58 ms)

Connecting to https://api.individual.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.113.21 (22 ms)
- DNS ipv6 Lookup: Error (16 ms): getaddrinfo ENOTFOUND api.individual.githubcopilot.com
- Proxy URL: None (2 ms)
- Electron fetch (configured): HTTP 200 (1016 ms)
- Node.js https: HTTP 200 (1108 ms)
- Node.js fetch: HTTP 200 (1212 ms)

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).