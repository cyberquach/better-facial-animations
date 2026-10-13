// The settings page (SKSE Menu Framework, when installed).
#ifndef INCLUDE_A2F_PLUGIN_MENU_H_
#define INCLUDE_A2F_PLUGIN_MENU_H_

#include "a2f/runtime/skse.h"

namespace bfa {

// Adds the page; call once every plugin is loaded.
void RegisterMenu();
void InitMenu(const skse::SKSEInterface* iface);
void MenuMessage(uint32_t type);

}  // namespace bfa

#endif  // INCLUDE_A2F_PLUGIN_MENU_H_
