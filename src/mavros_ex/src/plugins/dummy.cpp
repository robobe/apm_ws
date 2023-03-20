#include "rcpputils/asserts.hpp"
#include "mavros/mavros_uas.hpp"
#include "mavros/plugin.hpp"
#include "mavros/plugin_filter.hpp"

namespace mavros
{
namespace std_plugins
{

/**
 * @brief Dummy plugin.
 * @plugin dummy
 * @example_plugin
 *
 * Example and "how to" for users.
 */
class DummyExPlugin : public plugin::Plugin
{
public:

  explicit DummyExPlugin(plugin::UASPtr uas_)
  : Plugin(uas_, "dummy")
  {
  }


  Subscriptions get_subscriptions() override
  {
    return {
      /* automatic message deduction by second argument */
      make_handler(&DummyExPlugin::handle_heartbeat)
    };
  }

private:
  /**
   * This function will be called to handle HEARTBEAT from any source
   * when framing is ok.
   *
   * @param[in] msg     raw message frame
   * @param[in] hb      decoded message (require a type from mavlink c++11 library)
   * @param[in] filter  an instance of that filter will determine conditions to call that function
   */
  void handle_heartbeat(
    const mavlink::mavlink_message_t * msg [[maybe_unused]],
    mavlink::minimal::msg::HEARTBEAT & hb,
    plugin::filter::AnyOk filter [[maybe_unused]])
  {
    RCLCPP_INFO_STREAM(get_logger(), "Dummy::handle_heartbeat: " << hb.to_yaml());
  }

 
};

}       // namespace std_plugins
}       // namespace mavros

#include <mavros/mavros_plugin_register_macro.hpp>  // NOLINT
MAVROS_PLUGIN_REGISTER(mavros::std_plugins::DummyExPlugin)
