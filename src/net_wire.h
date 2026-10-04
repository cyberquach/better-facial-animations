// Messages between the game and the model's process (model_process.cc): a
// type byte, a 32-bit size and that many bytes. How the bytes travel is the
// caller's: read and write are callables (char*, size_t) -> bool that
// transfer all of them or fail.
#ifndef SRC_NET_WIRE_H_
#define SRC_NET_WIRE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace a2f {

// game -> process: one run, x (b, kWindow) then emo (b, kNumEmo) as floats;
// answered by kMsgOk, kMsgError or kMsgLost
constexpr char kMsgRun = 'R';
// process -> game
constexpr char kMsgHello = 'H';  // first: the process runs
constexpr char kMsgOk = 'K';     // the session's description / a run's output
constexpr char kMsgError = 'E';  // text: it failed
constexpr char kMsgLost = 'X';   // text: the run failed, the session is gone
constexpr char kMsgLog = 'L';    // text for the log, any time before an answer

// No message is larger: a size beyond it is not a message's.
constexpr uint32_t kMaxMsg = 16 << 20;

template <typename Write>
bool WriteMsg(Write&& write, char type, const void* data, size_t n) {
  std::string m(1 + sizeof(uint32_t) + n, type);
  const uint32_t size = static_cast<uint32_t>(n);
  memcpy(&m[1], &size, sizeof size);
  if (n) memcpy(&m[1 + sizeof size], data, n);
  return write(&m[0], m.size());
}

// False when the other end is gone or does not send messages.
template <typename Read>
bool ReadMsg(Read&& read, char& type, std::string& data) {
  char head[1 + sizeof(uint32_t)];
  uint32_t size;
  if (!read(head, sizeof head)) return false;
  type = head[0];
  memcpy(&size, head + 1, sizeof size);
  if (size > kMaxMsg) return false;
  data.resize(size);
  return !size || read(&data[0], size);
}

}  // namespace a2f

#endif  // SRC_NET_WIRE_H_
