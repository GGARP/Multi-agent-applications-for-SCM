import numpy as np
import gym
from gym import spaces

np.random.seed(0)

class DecentralizedInventoryEnvWithComm(gym.Env):
    """
    A decentralized inventory management environment with inter-agent communication.
    
    Each agent (stage) must decide how many units to order AND what communication
    message (a vector of size comm_size) to send out. The observation of each agent
    then includes other agents' messages from the previous time step, in addition to
    the usual inventory/backlog states and so on.
    """

    def __init__(
        self,
        num_stages,
        num_periods,
        init_inventories,
        lead_times,
        demand_fn,
        prod_capacities,
        sale_prices,
        order_costs,
        backlog_costs,
        holding_costs,
        stage_names,
        comm_size=2,  # dimension of communication
    ):
        super().__init__()

        # Store parameters
        self.num_stages = num_stages
        self.num_periods = num_periods
        self.init_inventories = np.array(init_inventories, dtype=int)
        self.lead_times = np.array(lead_times, dtype=int)
        self.max_lead_time = max(self.lead_times)
        self.demand_fn = demand_fn
        self.prod_capacities = np.array(prod_capacities, dtype=int)
        self.sale_prices = np.array(sale_prices, dtype=float)
        self.order_costs = np.array(order_costs, dtype=float)
        self.backlog_costs = np.array(backlog_costs, dtype=float)
        self.holding_costs = np.array(holding_costs, dtype=float)
        self.stage_names = stage_names
        self.comm_size = comm_size

        # Internal state variables
        self.period = 0
        self.inventories = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.orders = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.arriving_orders = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.sales = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.backlogs = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.demands = np.zeros(self.num_periods + 1, dtype=int)
        self.profits = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.total_profits = np.zeros(self.num_periods + 1, dtype=int)

        # We store the messages each agent sent in the previous step
        # Shape: (num_stages, comm_size)
        self.messages = None

        # ========== ACTION SPACE ==========
        # Each agent chooses:
        #  (1) an order quantity in {0, 1, ..., capacity} (discrete)
        #  (2) a communication vector of size comm_size (continuous)
        self.single_agent_action_space = spaces.Tuple((
            spaces.Discrete(np.max(self.prod_capacities) + 1),
            spaces.Box(low=-1.0, high=1.0, shape=(comm_size,), dtype=np.float32),
        ))
        # For the multi-agent setting, we can represent the total action space as
        # num_stages copies of the single-agent action space.
        # (Alternatively, you can flatten these if you prefer.)
        self.action_space = spaces.Tuple(
            [self.single_agent_action_space for _ in range(self.num_stages)]
        )

        # ========== OBSERVATION SPACE ==========
        # We keep the original dimension (9 + 2 * max_lead_time) for each stage,
        # and now we add space for communication messages from all agents.
        #
        # In this example, each agent sees:
        #   - the original environment state (9 + 2 * max_lead_time)
        #   - its own last message (comm_size)
        #   - the other agents' last messages (num_stages - 1) * comm_size
        #
        # So the total dimension for each stage is:
        #   obs_dim = 9 + 2*max_lead_time + num_stages * comm_size
        # We will create a Box space for each stage with these dimensions.
        obs_dim_per_stage = 9 + 2 * self.max_lead_time + self.num_stages * self.comm_size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_stages, obs_dim_per_stage),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.period = 0
        self.inventories.fill(0)
        self.orders.fill(0)
        self.arriving_orders.fill(0)
        self.sales.fill(0)
        self.backlogs.fill(0)
        self.demands.fill(0)
        self.profits.fill(0)
        self.total_profits.fill(0)

        # Initialize the messages to zero
        self.messages = np.zeros((self.num_stages, self.comm_size), dtype=np.float32)

               # Set the initial condition and state
        self.inventories[:, 0] = self.init_inventories
        self.update_state()

        return self.state_dict, {}
    def update_state(self) -> None:
        """
        Update the environment state including the current stage features, inventory, backlog, upstream backlog,
        previous sales, arriving deliveries, and communication messages.

        Original State: s_{m,t} = [c_m, p_m, r_m, k_m, h_m, L_m, I_{m,t-1}, B_{m,t-1}, B_{m+1,t-1},
                         S_{m,t-L_max}, ..., S_{m,t-1}, 0, ..., 0, R_{m,t-L_m}, ..., R_{m,t-1}]
        Modified State: s_{m,t} = [ ...original features..., self.messages[m]]
        """
        t = self.period
        # Compute the base state without communication messages.
        base_states = np.zeros((self.num_stages, 9 + 2 * self.max_lead_time), dtype=int)
        base_states[:, :8] = np.stack([
            self.prod_capacities, self.sale_prices, self.order_costs, self.backlog_costs, self.holding_costs,
            self.lead_times, self.inventories[:, t], self.backlogs[:, t]], axis=-1)
        base_states[:-1, 8] = self.backlogs[1:, t]

        lt_max = self.max_lead_time
        if t >= lt_max:
            base_states[:, (-2 * lt_max):-lt_max] = self.sales[:, (t - lt_max + 1):(t + 1)]
        elif t > 0:
            base_states[:, (-lt_max - t):-lt_max] = self.sales[:, 1:(t + 1)]

        for m in range(self.num_stages):
            lt = self.lead_times[m]
            if t >= lt:
                base_states[m, -lt:] = self.arriving_orders[m, (t - lt + 1):(t + 1)]
            elif t > 0:
                base_states[m, -t:] = self.arriving_orders[m, 1:(t + 1)]

        # Incorporate communication messages from ALL agents.
        # Flatten the messages array (shape: [num_stages, comm_size])
        # so that each stage's state gets all agents' messages.
        all_messages = self.messages.flatten()  # shape becomes (self.num_stages * comm_size,)
        # Repeat this vector for each stage.
        repeated_messages = np.tile(all_messages, (self.num_stages, 1))
        
        # Concatenate the base state with the repeated communication messages.
        new_states = np.concatenate([base_states, repeated_messages], axis=-1)

        # Update the state dictionary for each stage.
        self.state_dict = {f"stage_{m}": new_states[m] for m in range(self.num_stages)}
        
        
    def step(self, action_dict: dict[str, tuple]) -> tuple[dict, dict, dict, dict, dict]:
        """
        Take a step and return the next observation.

        :param action_dict: action for each stage, where each action is a tuple:
                            (order_quantity, communication_vector)
        :return: states, rewards, terminations, truncations, infos
        """
        # (Optional) Add assertions to check that each action is a tuple and order_quantity is non-negative.
        assert all(f"stage_{m}" in action_dict for m in range(self.num_stages)), \
            "Actions for all stages are required."
        assert all(action_dict[f"stage_{m}"][0] >= 0 for m in range(self.num_stages)), \
            "Order quantities must be non-negative integers."

        # Update period and set time index
        self.period += 1
        t = self.period
        M = self.num_stages

                # --- NEW: Process actions to update orders and messages ---
        orders_array = np.zeros(self.num_stages, dtype=int)
        new_messages = np.zeros((self.num_stages, self.comm_size), dtype=np.float32)
        for stage in range(self.num_stages):
            order_quantity, comm_vector = action_dict[f"stage_{stage}"]
            clipped_order = np.clip(order_quantity, 0, self.prod_capacities[stage])
            orders_array[stage] = clipped_order

            # Update the communication vector so that the order quantity element reflects the clipped order.
            comm_vector[1] = float(clipped_order)

            new_messages[stage] = comm_vector

        self.orders[:, t] = orders_array
        self.messages = new_messages
        

        # Get the inventory at the beginning of the period
        current_inventories = self.inventories[:, t - 1]
        self.demands[t] = int(self.demand_fn(t))

        # Add the delivered orders
        for m in range(self.num_stages):
            lt = self.lead_times[m]
            if t >= lt:
                current_inventories[m] += self.arriving_orders[m, t - lt]

        # Compute the fulfilled orders
        self.arriving_orders[:-1, t] = np.minimum(
            np.minimum(self.backlogs[1:, t - 1] + self.orders[:-1, t], current_inventories[1:]),
            self.prod_capacities[1:])
        self.arriving_orders[M - 1, t] = self.orders[M - 1, t]

        # Compute the sales
        self.sales[1:, t] = self.arriving_orders[:-1, t]
        self.sales[0, t] = min(
            min(self.backlogs[0, t - 1] + self.demands[t], current_inventories[0]),
            self.prod_capacities[0])

        # Compute the backlogs
        self.backlogs[1:, t] = self.backlogs[1:, t - 1] + self.orders[:-1, t] - self.sales[1:, t]
        self.backlogs[0, t] = self.backlogs[0, t - 1] + self.demands[t] - self.sales[0, t]

        # Compute the inventory at the end of the period
        self.inventories[:, t] = current_inventories - self.sales[:, t]

        # Compute the profits
        self.profits[:, t] = self.sale_prices * self.sales[:, t] - self.order_costs * self.arriving_orders[:, t] \
                             - self.backlog_costs * self.backlogs[:, t] - self.holding_costs * self.inventories[:, t]
        self.total_profits[t] = np.sum(self.profits[:, t])

        # Determine rewards and terminations
        rewards = {f"stage_{m}": self.profits[m, t] for m in range(self.num_stages)}
        all_termination = self.period >= self.num_periods
        terminations = {f"stage_{m}": all_termination for m in range(self.num_stages)}
        terminations["__all__"] = all_termination
        truncations = {f"stage_{m}": False for m in range(self.num_stages)}
        truncations["__all__"] = False
        infos = {f"stage_{m}": {} for m in range(self.num_stages)}

        # Update the state
        self.update_state()

        return self.state_dict, rewards, terminations, truncations, infos

        
    def _parse_state(self, state: list) -> dict:
        """
        Parse a single stage state, now including the communication messages.

        :param state: state
        :return: parsed state
        """
        lt_max = self.max_lead_time
        base_dim = 9 + 2 * lt_max  # original base state dimension

        # Extract the base state features
        base = state[:base_dim]
        communications = state[base_dim:]  # remaining elements are communications

        return {
            'prod_capacity': base[0],
            'sale_price': base[1],
            'order_cost': base[2],
            'backlog_cost': base[3],
            'holding_cost': base[4],
            'lead_time': base[5],
            'inventory': base[6],
            'backlog': base[7],
            'upstream_backlog': base[8],
            'sales': base[(-2 * lt_max):(-lt_max)].tolist(),
            'deliveries': base[-lt_max:].tolist(),
            'communications': communications  # returns full flattened communication vector
    }
    def parse_state(self, state_dict: dict = None) -> dict:
        """
        Parse the state dictionary

        :param state_dict: state dictionary
        :return: parsed state dict
        """
        if state_dict is None:
            state_dict = self.state_dict

        parsed_state = {}

        for stage_id_name, state in state_dict.items():
            parsed_state[stage_id_name] = self._parse_state(state)

        return parsed_state


        

    def render(self, mode="human"):
        if mode == "human":
            print(f"Period: {self.period}")
            for stage in range(self.num_stages):
                print(
                    f"Stage {stage} ({self.stage_names[stage]}): "
                    f"Inventory={self.inventories[stage]}, "
                    f"Backlog={self.backlogs[stage]}, "
                    f"Last Message={self.messages[stage]}"
                )
def env_creator(env_config):
    """
    Create the environment
    """
    if env_config is None:
        env_config = env_configs['two_agent']

    return DecentralizedInventoryEnvWithComm(
        num_stages=env_config['num_stages'],
        num_periods=env_config['num_periods'],
        init_inventories=env_config['init_inventories'],
        lead_times=env_config['lead_times'],
        demand_fn=env_config['demand_fn'],
        prod_capacities=env_config['prod_capacities'],
        sale_prices=env_config['sale_prices'],
        order_costs=env_config['order_costs'],
        backlog_costs=env_config['backlog_costs'],
        holding_costs=env_config['holding_costs'],
        stage_names=env_config['stage_names'],
    )


if __name__ == '__main__':
    im_env = env_creator(env_configs['two_agent'])
    im_env.reset()
    print(f"stage_names = {im_env.stage_names}")
    print(f"state_dict = {im_env.state_dict}")
    print(f"state_dict = {im_env.parse_state(im_env.state_dict)}")
    print(f"observation_space = {im_env.observation_space}")
    print(f"observation_sample = {im_env.observation_space.sample()}")
    print(f"action_space = {im_env.action_space}")
    print(f"action_sample = {im_env.action_space.sample()}")

    for t in range(im_env.num_periods):
        next_state_dict, rewards, terminations, truncations, infos = im_env.step(
            action_dict={f"stage_{m}": 4 for m in range(im_env.num_stages)})
        print('-' * 80)
        print(f"period = {t}")
        print(f"next_state_dict = {next_state_dict}")
        print(f"next_state_dict = {im_env.parse_state(next_state_dict)}")
        print(f"rewards = {rewards}")
        print(f"terminations = {terminations}")
        print(f"truncations = {truncations}")
        print(f"infos = {infos}") 