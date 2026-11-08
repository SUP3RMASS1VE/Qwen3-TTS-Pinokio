const fs = require("fs")
const path = require("path")

// app/venv is created before dependencies install, so an interrupted or failed
// install would still look installed; require gradio to actually be present.
const depsInstalled = () => {
  const venv = path.join(__dirname, "app", "venv")
  if (fs.existsSync(path.join(venv, "Lib", "site-packages", "gradio"))) return true
  try {
    return fs.readdirSync(path.join(venv, "lib")).some((d) => fs.existsSync(path.join(venv, "lib", d, "site-packages", "gradio")))
  } catch {
    return false
  }
}

module.exports = {
  version: "5.0",
  title: "Qwen3-TTS",
  description: "Qwen3-TTS is an open-source series of TTS models developed by the Qwen team",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = depsInstalled()
    let running = {
      install: info.running("install.json"),
      start: info.running("start.json"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.json",
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.json")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.json",
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.json",
          }]
        }
      } else if (running.update) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Updating",
          href: "update.js",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Resetting",
          href: "reset.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.json",
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.json",
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
          href: "reset.js",
          confirm: "Are you sure you wish to reset the app?"

        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.json",
      }]
    }
  }
}


